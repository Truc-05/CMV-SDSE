"""Motivation figure: why 'add more' / 'add a different verifier' is not enough.

nd.md's motivation figure (\\label{fig:motivation}) is a single three-panel float that answers
the question in one glance, cherry-picking the load-bearing panel from each observation:
  (a) figures/fig_sat_a.png -- same-model n_eff(k) saturates near 1 (count != evidence)
  (b) figures/fig_div_b.png -- dependence vs conditional residual catch (diversity != value)
  (c) figures/fig_div_c.png -- the conditional marginal catch the controller actually gates on
All three panels are produced by scripts/make_figures.py (group_saturation / group_diversity)
and already carry their own "(a)/(b)/(c)" titles, so this only scales them to a common height
and concatenates. The full saturation and diversity figures remain in the appendix.
Writes figures/fig_motivation.{png,pdf} (the .pdf is what nd.md includes).
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "figures"
PANELS = ["fig_sat_a.png", "fig_div_b.png", "fig_div_c.png"]
DPI = 170.0


def main():
    ims = [Image.open(OUT / p).convert("RGB") for p in PANELS]
    h = min(i.height for i in ims)
    ims = [i.resize((round(i.width * h / i.height), h)) for i in ims]
    w = sum(i.width for i in ims)
    canvas = Image.new("RGB", (w, h), "white")
    x = 0
    for i in ims:
        canvas.paste(i, (x, 0))
        x += i.width
    canvas.save(OUT / "fig_motivation.png")
    canvas.save(OUT / "fig_motivation.pdf", "PDF", resolution=DPI)
    print(f"wrote figures/fig_motivation.{{png,pdf}}  {canvas.size}px  aspect={w/h:.2f}")


if __name__ == "__main__":
    main()
