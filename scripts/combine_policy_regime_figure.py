"""Combine the decision-map and fusion-regime panels into one paper figure.

nd.md's Figure 4 (\\label{fig:policy_regime}) is a single two-panel float:
  (a) figures/fig_landscape.png -- CMV-SDSE Scale/Diversify/Stop/Escalate decision map
  (b) figures/fig_regime.png    -- operating-regime-dependent fusion (U vs cost ratio rho)
Both panels are produced by scripts/make_figures.py (schematic_landscape / single_regime)
at an identical 1258x850. This script montages them left->right, stamps "(a)"/"(b)" in the
top-left corners, and writes figures/fig_policy_regime.{png,pdf} (the .pdf is what nd.md
includes). Re-run make_figures.py first if the source panels changed.
"""
from __future__ import annotations
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

OUT = Path(__file__).resolve().parents[1] / "figures"
DPI = 170.0  # matches make_figures.py savefig.dpi


def main():
    a = Image.open(OUT / "fig_landscape.png").convert("RGB")  # (a) decision map
    b = Image.open(OUT / "fig_regime.png").convert("RGB")     # (b) regime
    if a.size != b.size:
        raise SystemExit(f"panel size mismatch: {a.size} vs {b.size} -- regen make_figures.py")
    w, h = a.size
    canvas = Image.new("RGB", (2 * w, h), "white")
    canvas.paste(a, (0, 0))
    canvas.paste(b, (w, 0))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold")), 34)
    draw.text((16, 8), "(a)", fill="#2b2b2b", font=font)
    draw.text((w + 16, 8), "(b)", fill="#2b2b2b", font=font)

    canvas.save(OUT / "fig_policy_regime.png")
    canvas.save(OUT / "fig_policy_regime.pdf", "PDF", resolution=DPI)
    print(f"wrote figures/fig_policy_regime.{{png,pdf}}  {canvas.size}px "
          f"(~{canvas.size[0]/DPI:.1f}x{canvas.size[1]/DPI:.1f} in @ {DPI:g} dpi)")


if __name__ == "__main__":
    main()
