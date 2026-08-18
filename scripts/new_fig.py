High-fidelity 4-figure template for CMV-SDSE paper.

One script -> four independent figures (PNG + PDF):
  A. Residual Error Recovery (6 panels)
  B. Calibration -> Frozen Test Generalization (4 panels)
  C. Residual Error Overlap Atlas (6 panels)
  D. Joint Policy Atlas (6 panels)

IMPORTANT:
- Values below are TEMPLATE/PLACEHOLDER values reconstructed from the visual mockup.
- Replace DATA_* blocks with your frozen manuscript results before submission.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon
from matplotlib.lines import Line2D

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# -----------------------------
# Visual constants
# -----------------------------
NAVY = "#17306f"
BLUE = "#4d8ed8"
BLUE2 = "#73b5df"
PURPLE = "#6f45bd"
GREEN = "#78a85a"
PALE_GREEN = "#dcefcf"
RED = "#e4483d"
ORANGE = "#f4a340"
GRAY = "#aaa7a0"
LIGHT_GRAY = "#e7e6e3"
INK = "#111111"
GRID = "#d8d8d8"
BORDER = "#7887dd"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 350,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FAMILIES = ["ARC", "MMLU", "CSQA", "TruthfulQA", "GSM8K", "MBPP"]
TYPE = ["MCQA", "MCQA", "MCQA", "Natural", "Natural", "Natural"]

# -------------------------------------------------------------------
# DATA A — placeholder/template values from the supplied visual mockup
# -------------------------------------------------------------------
DATA_A = {
    "ARC":        dict(total=150, solo=110, residual=40, second=27, third=8, missed=5,  neff=1.12, div=0.38, gain=23.3),
    "MMLU":       dict(total=150, solo=107, residual=43, second=29, third=10, missed=4, neff=1.05, div=0.31, gain=27.9),
    "CSQA":       dict(total=150, solo=100, residual=50, second=32, third=12, missed=6, neff=1.41, div=0.52, gain=32.9),
    "TruthfulQA": dict(total=150, solo=84,  residual=66, second=31, third=17, missed=18, neff=1.71, div=0.67, gain=36.4),
    "GSM8K":      dict(total=150, solo=111, residual=39, second=18, third=7, missed=14, neff=1.08, div=0.29, gain=19.5),
    "MBPP":       dict(total=150, solo=106, residual=44, second=22, third=10, missed=12, neff=1.26, div=0.41, gain=25.6),
}

# -----------------------------
# DATA B — template values
# -----------------------------
CAL_U  = np.array([0.47, 0.72, 0.66, 0.59, 0.54, 0.52])
TEST_U = np.array([0.84, 0.84, 0.86, 0.90, 0.85, 0.97])

CMV_MINUS_GES = np.array([0.06, 0.02, 0.04, 0.02, -0.025, -0.04])
CMV_ERR = np.array([0.055, 0.06, 0.06, 0.045, 0.05, 0.055])

CAL_SIZE  = np.array([1.0, 1.2, 1.35, 1.0, 1.2, 1.2])
TEST_SIZE = np.array([2.0, 2.35, 2.2, 2.55, 2.85, 3.0])

REG_FIXED = np.array([0.008, 0.015, 0.014, 0.018, 0.012, 0.017])
REG_TOPO  = np.array([0.045, 0.048, 0.050, 0.052, 0.055, 0.058])
REG_JOINT = np.array([0.007, 0.010, 0.011, 0.013, 0.009, 0.012])

# -----------------------------
# DATA C — template UpSet counts
# -----------------------------
UPSET_COUNTS = {
    "ARC":        [31,22,16,11,8,6,5,4,3,2,2,1,1,1,1],
    "MMLU":       [28,19,10,7,6,5,4,3,2,2,1,1,1,1,1],
    "CSQA":       [29,21,15,10,7,5,5,4,3,2,2,1,1,1,1],
    "TruthfulQA": [27,20,16,8,7,5,5,3,2,2,2,1,1,1,1],
    "GSM8K":      [33,18,13,7,6,5,4,3,2,2,2,1,1,1,1],
    "MBPP":       [30,17,15,10,7,5,5,3,2,2,2,1,1,1,1],
}
SOURCES = ["Qwen-7B", "Mistral-7B", "Llama3.1-8B", "Gemma2-9B", "Phi-3.5-3B"]

# -----------------------------
# DATA D — template policy atlas
# -----------------------------
RHO = [0, .25, .5, 1, 2, 4]
POLICY = {
    "ARC":        dict(top=["OR","OR","OR","OR","OR","AND"], size=[1,1,1,1.1,1.3,1.8], pick=["Qwen"]*5+["Mistral"]),
    "MMLU":       dict(top=["OR","OR","OR","OR","AND","AND"], size=[1,1,1.1,1.3,1.8,2.6], pick=["Qwen"]*5+["Mistral"]),
    "CSQA":       dict(top=["OR","OR","OR","OR","AND","AND"], size=[1,1,1.1,1.2,1.7,2.4], pick=["Qwen"]*5+["Mistral"]),
    "TruthfulQA": dict(top=["OR","OR","OR","OR","AND","AND"], size=[1,1,1.2,1.4,2.1,3.0], pick=["Mistral"]*4+["Qwen","Qwen"]),
    "GSM8K":      dict(top=["OR"]*6, size=[1,1,1,1.1,1.2,1.6], pick=["Llama3.1"]*6),
    "MBPP":       dict(top=["OR","OR","OR","OR","AND","AND"], size=[1,1,1,1.2,1.6,2.2], pick=["Mistral"]*4+["Qwen","Qwen"]),
}

# -----------------------------
# Helpers
# -----------------------------
def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

def outer_frame(fig):
    fig.add_artist(Rectangle((0.004, 0.006), .992, .988,
                             transform=fig.transFigure, fill=False,
                             lw=.8, ec=BORDER))

def ribbon(ax, x0, x1, y0, y1, h, color, alpha=.72):
    """Simple tapered ribbon polygon."""
    poly = Polygon([
        (x0, y0-h/2), (x0, y0+h/2),
        (x1, y1+h*.32), (x1, y1-h*.32)
    ], closed=True, fc=color, ec="none", alpha=alpha)
    ax.add_patch(poly)

# ================================================================
# FIGURE A
# ================================================================
def figure_A():
    fig, axes = plt.subplots(1, 6, figsize=(16.5, 3.15))
    fig.subplots_adjust(left=.035, right=.87, top=.79, bottom=.19, wspace=.24)
    fig.suptitle("Figure A. Residual Error Recovery — What CMV Actually Buys (6 benchmarks)",
                 fontsize=15, fontweight="bold", y=.96, color="#111438")

    cols = [NAVY, GRAY, GREEN, PURPLE, RED]
    labels = ["Caught by best solo", "Residual after best solo",
              "Recovered by 2nd (CMV pick)", "Recovered by 3rd (CMV pick)",
              "Still missed (unresolved)"]

    for i, (ax, fam, typ) in enumerate(zip(axes, FAMILIES, TYPE)):
        d = DATA_A[fam]
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.set_title(f"({chr(97+i)}) {fam} ({typ})", fontweight="bold", pad=6)

        # left source block
        ax.add_patch(Rectangle((.04,.18), .09,.68, fc="#f5f6f8", ec="#777", lw=.6))
        ax.text(.005,.52, f"{d['total']}\ntotal\nerrors", ha="right", va="center",
                fontsize=8, fontweight="bold")

        vals = [d["solo"], d["residual"], d["second"], d["third"], d["missed"]]
        ys = [.80,.62,.47,.34,.23]
        heights = np.array(vals)/d["total"]*.27 + .025

        for val, y, h, c, lab in zip(vals, ys, heights, cols, labels):
            ribbon(ax, .13, .73, y, y, h, c)
            ax.text(.16, y, lab, va="center", fontsize=6.5,
                    color="white" if c in [NAVY, PURPLE, RED] else INK)
            ax.text(.77, y, f"{val}", va="center", fontsize=8, fontweight="bold")

        # metric card
        card = FancyBboxPatch((.01,-.03), .96,.16, boxstyle="round,pad=.012,rounding_size=.025",
                              transform=ax.transAxes, fc="white", ec="#bbb", lw=.65, clip_on=False)
        ax.add_patch(card)
        for x in [.33,.66]:
            ax.plot([x,x],[-.03,.13], transform=ax.transAxes, color="#ddd", lw=.7, clip_on=False)
        ax.text(.17,.08,r"$n_{\rm eff}(50)$",ha="center",transform=ax.transAxes,fontsize=7)
        ax.text(.50,.08,"Diversity (JSD)",ha="center",transform=ax.transAxes,fontsize=7)
        ax.text(.83,.08,"CMV gain",ha="center",transform=ax.transAxes,fontsize=7)
        ax.text(.17,.00,f"{d['neff']:.2f}",ha="center",transform=ax.transAxes,fontsize=9,fontweight="bold")
        ax.text(.50,.00,f"{d['div']:.2f}",ha="center",transform=ax.transAxes,fontsize=9,fontweight="bold")
        ax.text(.83,.00,f"{d['gain']:.1f}%",ha="center",transform=ax.transAxes,fontsize=9,fontweight="bold")

    # legend + definition box
    leg_ax = fig.add_axes([.875,.36,.115,.38]); leg_ax.axis("off")
    handles = [Rectangle((0,0),1,1,fc=c) for c in cols]
    leg_ax.legend(handles, labels, loc="upper left", frameon=True, fontsize=7,
                  handlelength=1.7, borderpad=.7)
    fig.text(.882,.18,
             "CMV gain = errors recovered\nby CMV selected sources /\nresidual after best solo",
             fontsize=7.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=.5", fc="white", ec="#888", ls="--", lw=.7))
    outer_frame(fig)
    save(fig, "figure_A_residual_recovery")

# ================================================================
# FIGURE B
# ================================================================
def figure_B():
    fig, axes = plt.subplots(1,4,figsize=(16.5,4.1))
    fig.subplots_adjust(left=.06,right=.985,top=.78,bottom=.17,wspace=.32)
    fig.suptitle("Figure B. Calibration → Frozen Test Generalization (4 panels)",
                 fontsize=15,fontweight="bold",y=.96,color="#111438")
    y = np.arange(len(FAMILIES))

    # a: paired dumbbell
    ax=axes[0]
    for j in y:
        ax.plot([CAL_U[j],TEST_U[j]],[j,j],ls="--",lw=1,color="#888")
    ax.scatter(CAL_U,y,s=38,c=BLUE,edgecolors=INK,lw=.6,label="Calibration",zorder=3)
    ax.scatter(TEST_U,y,s=38,c=ORANGE,edgecolors=INK,lw=.6,label="Frozen Test",zorder=3)
    ax.set_yticks(y,FAMILIES); ax.invert_yaxis(); ax.set_xlim(.35,1.02)
    ax.set_xlabel("Utility U"); ax.set_title("(a) Calibration Utility → Frozen Test Utility ($\\rho=1$)",fontweight="bold")
    ax.grid(axis="x",color=GRID,lw=.5,alpha=.6); ax.legend(frameon=False,fontsize=7,loc="upper left")
    ax.text(.97,.16,"Spearman\n$\\rho=0.91$\n$(p=0.004)$",transform=ax.transAxes,ha="right",fontsize=8)

    # b: forest/difference
    ax=axes[1]
    ax.axvline(0,color=INK,lw=1)
    for j in y:
        ax.errorbar(CMV_MINUS_GES[j],j,xerr=CMV_ERR[j],fmt="o",ms=5,
                    color="#2b9a92",ecolor="#2b6f78",capsize=2)
    ax.set_yticks(y,FAMILIES); ax.invert_yaxis(); ax.set_xlim(-.16,.16)
    ax.set_xlabel(r"$\Delta$ Utility"); ax.set_title("(b) CMV-SDSE vs. GES (Frozen Test, $\\rho=1$)",fontweight="bold")
    ax.grid(axis="x",color=GRID,lw=.5,alpha=.6)
    ax.text(.98,.28,"+ better\n(CMV)",transform=ax.transAxes,ha="right",color=GREEN,fontweight="bold")
    ax.text(.98,.06,"Sign test\n$p=0.031$",transform=ax.transAxes,ha="right",fontsize=8)

    # c: selected set size
    ax=axes[2]
    for j in y:
        ax.plot([CAL_SIZE[j],TEST_SIZE[j]],[j,j],ls="--",lw=1,color="#888")
    ax.scatter(CAL_SIZE,y,s=38,c=BLUE,edgecolors=INK,lw=.6,label="Calibration",zorder=3)
    ax.scatter(TEST_SIZE,y,s=38,c=ORANGE,edgecolors=INK,lw=.6,label="Frozen Test",zorder=3)
    ax.set_yticks(y,FAMILIES); ax.invert_yaxis(); ax.set_xlim(.75,3.2)
    ax.set_xlabel(r"$|S|$"); ax.set_title("(c) Selected Set Size $|S|$ ($\\rho=1$)",fontweight="bold")
    ax.grid(axis="x",color=GRID,lw=.5,alpha=.6); ax.legend(fontsize=7,loc="upper right")

    # d: regret
    ax=axes[3]
    ax.plot(REG_FIXED,y,"o-",lw=1,c=PURPLE,label="Fixed-OR")
    ax.plot(REG_TOPO,y,"o-",lw=1,c="#29956f",label="Topology-only")
    ax.plot(REG_JOINT,y,"o-",lw=1,c=BLUE,label="Joint (CMV-SDSE)")
    ax.set_yticks(y,FAMILIES); ax.invert_yaxis(); ax.set_xlim(-.002,.11)
    ax.set_xlabel("Regret"); ax.set_title("(d) Regret vs. Joint Optimum (Frozen Test)",fontweight="bold")
    ax.grid(axis="x",color=GRID,lw=.5,alpha=.6); ax.legend(fontsize=7,loc="upper right")
    ax.text(.97,.18,"Joint dominates\nacross benchmarks",transform=ax.transAxes,ha="right",fontsize=8)

    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
    outer_frame(fig)
    save(fig,"figure_B_generalization")

# ================================================================
# FIGURE C
# ================================================================
def draw_upset(ax, fam, typ):
    counts = UPSET_COUNTS[fam]
    n = len(counts)
    # deterministic combination patterns
    patterns = [
        [0],[1],[2],[3],[4],[0,1],[0,2],[1,2],[1,3],[2,3],
        [0,1,2],[1,2,3],[0,3],[2,4],[0,1,2,3]
    ]
    ax.set_xlim(-1.2,n-.3); ax.set_ylim(-5.7,max(counts)*1.35)
    ax.axis("off")
    ax.set_title(f"{fam} ({typ})",fontweight="bold",fontsize=9,pad=3)
    # bars
    for x,c in enumerate(counts):
        ax.add_patch(Rectangle((x-.32,0),.64,c,fc=NAVY,alpha=.72))
        ax.text(x,c+1,f"{c}",ha="center",va="bottom",fontsize=6,color=NAVY)
    # matrix
    for x,pat in enumerate(patterns):
        for r in range(5):
            ax.scatter(x,-1-r,s=12,c="#dedede",zorder=1)
        ys=[-1-r for r in pat]
        if len(ys)>1: ax.plot([x,x],[min(ys),max(ys)],c=NAVY,lw=1.1,zorder=2)
        ax.scatter([x]*len(ys),ys,s=15,c=NAVY,zorder=3)
    # source labels only on leftmost-ish, still readable each panel
    for r,s in enumerate(SOURCES):
        ax.text(-.7,-1-r,s,ha="right",va="center",fontsize=5.8)
    ax.text(-.8,max(counts)*.7,"Intersection\nsize",ha="right",fontsize=5.7)
    ax.text(n-.2,max(counts)*1.15,"150 errors",ha="right",fontsize=5.7)

def figure_C():
    fig,axes=plt.subplots(2,3,figsize=(16.5,6.2))
    fig.subplots_adjust(left=.065,right=.985,top=.84,bottom=.16,wspace=.25,hspace=.42)
    fig.suptitle("Figure C. Residual Error Overlap Atlas — Dependence & Complementarity (6 benchmarks)",
                 fontsize=15,fontweight="bold",y=.97,color="#111438")
    for ax,fam,typ in zip(axes.flat,FAMILIES,TYPE):
        draw_upset(ax,fam,typ)

    handles=[
        Rectangle((0,0),1,1,fc=NAVY,alpha=.75),
        Rectangle((0,0),1,1,fc=BLUE2,alpha=.8)
    ]
    fig.legend(handles,
        ["ERRORS missed by the above set of sources (intersection) → dependence",
         "ERRORS uniquely missed (only by that source) → complementarity"],
        loc="lower left",bbox_to_anchor=(.07,.035),ncol=1,fontsize=7,
        frameon=True,edgecolor=BORDER)
    outer_frame(fig)
    save(fig,"figure_C_overlap_atlas")

# ================================================================
# FIGURE D
# ================================================================
def draw_policy(ax,fam,typ):
    d=POLICY[fam]
    ax.set_xlim(-.2,6); ax.set_ylim(-.05,4.1); ax.axis("off")
    ax.set_title(f"{fam} ({typ})",fontweight="bold",fontsize=9,pad=3)

    # rho labels
    for i,r in enumerate(RHO):
        ax.text(i+.5,3.88,f"{r:g}",ha="center",fontsize=6.5)
    ax.text(-.15,3.88,"Cost regime $\\rho$",ha="right",fontsize=6.3)

    # topology row
    ax.text(-.15,3.15,"Topology",ha="right",va="center",fontsize=6.5,fontweight="bold")
    for i,t in enumerate(d["top"]):
        c=BLUE if t=="OR" else PURPLE
        ax.add_patch(Rectangle((i,2.75),1,.65,fc=c,ec="white",lw=.5))
        ax.text(i+.5,3.075,t,ha="center",va="center",fontsize=6.3,color="white")

    # size row
    ax.text(-.15,2.20,r"$|S|$ (sources)",ha="right",va="center",fontsize=6.5,fontweight="bold")
    for i,s in enumerate(d["size"]):
        ax.add_patch(Rectangle((i,1.88),1,.62,fc="white",ec="#ddd",lw=.5))
        ax.text(i+.5,2.19,f"{s:g}",ha="center",va="center",fontsize=6.5)

    # first pick
    ax.text(-.15,1.28,"First pick",ha="right",va="center",fontsize=6.5,fontweight="bold")
    for i,p in enumerate(d["pick"]):
        ax.add_patch(Rectangle((i,.96),1,.62,fc=PALE_GREEN,ec="white",lw=.5))
        ax.text(i+.5,1.27,p,ha="center",va="center",fontsize=5.7)

    # action mix decorative but informative template
    ax.text(-.15,.40,"Action mix",ha="right",va="center",fontsize=6.5,fontweight="bold")
    mix = [RED, BLUE, BLUE, BLUE2, BLUE2, ORANGE]
    if d["top"][-1]=="AND":
        mix[-2]=PURPLE
    for i,c in enumerate(mix):
        ax.add_patch(Rectangle((i,.12),1,.52,fc=c,ec="none",alpha=.88))

def figure_D():
    fig,axes=plt.subplots(2,3,figsize=(16.5,6.2))
    fig.subplots_adjust(left=.075,right=.985,top=.84,bottom=.15,wspace=.24,hspace=.42)
    fig.suptitle("Figure D. Joint Policy Atlas — How Optimal Policy Transitions with Cost Regime",
                 fontsize=15,fontweight="bold",y=.97,color="#111438")
    for ax,fam,typ in zip(axes.flat,FAMILIES,TYPE):
        draw_policy(ax,fam,typ)

    handles=[
        Rectangle((0,0),1,1,fc=BLUE),
        Rectangle((0,0),1,1,fc=BLUE2),
        Rectangle((0,0),1,1,fc=PURPLE),
        Rectangle((0,0),1,1,fc=RED),
        Rectangle((0,0),1,1,fc=ORANGE),
    ]
    fig.legend(handles,["OR (singleton)","OR (multi-source)","AND (multi-source)","STOP","ESCALATE (fallback)"],
               loc="lower center",bbox_to_anchor=(.5,.025),ncol=5,fontsize=7,
               frameon=True,edgecolor=BORDER)
    outer_frame(fig)
    save(fig,"figure_D_joint_policy_atlas")

if __name__ == "__main__":
    figure_A()
    figure_B()
    figure_C()
    figure_D()
    print(f"Done. Files written to: {OUT.resolve()}")