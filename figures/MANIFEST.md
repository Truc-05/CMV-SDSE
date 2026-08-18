# Figures — grouped panels (you combine each group into one composite)

Regenerate all panels:  `~/venvs/env/bin/python scripts/make_figures.py`
Panels within a group share size + style. This script does NOT montage — you combine
each group's a/b/c(/d/e/f) into `figures/<combined>.png` (the name pp.md references).

## GROUPS to combine  (panels a → f)
| combine these panels | into (pp.md uses) | panels | what each panel is |
|---|---|---|---|
| `fig_dmap_a … fig_dmap_f` | `fig_decision_maps.png` | **6** | controller (λ,ρ) decision map, ONE PER FAMILY: (a)ARC (b)MMLU (c)CSQA (d)TruthfulQA (e)GSM8K (f)MBPP |
| `fig_sat_a,b,c`   | `fig_saturation.png`     | 3 | (a) n_eff(k) all 7 families (b) reliability(k)+band (c) false-alarm(k)+band |
| `fig_div_a,b,c`   | `fig_obs2_diversity.png` | 3 | (a) count≠value (b) diversity≠value (c) selection gate CI-low>0 |
| `fig_seven_a,b,c` | `fig_sevenfamily.png`    | 3 | (a) n_eff@50 (b) ρ_v (c) miss — all 7 families |
| `fig_blind_a,b,c` | `fig_blindspot.png`      | 3 | (a) π₀ (b) power vs n (c) raw-vs-atom — (optional, not yet cited in pp.md) |
| `fig_landscape` + `fig_regime` | `fig_policy_regime.{png,pdf}` | 2 | (a) decision map (b) operating-regime fusion — `scripts/combine_policy_regime_figure.py` (nd.md Fig. `fig:policy_regime`) |
| `fig_sat_a` + `fig_div_b` + `fig_div_c` | `fig_motivation.{png,pdf}` | 3 | (a) n_eff saturation (b) diversity≠value (c) selection gate — `scripts/combine_motivation_figure.py` (nd.md Fig. `fig:motivation`; full `fig_saturation`/`fig_obs2_diversity` kept in appendix) |

## SINGLES  (already one file — no combining needed; pp.md uses these names directly)
`fig_architecture.png` · `fig_significance.png` (2 subplots) · `fig_beforeafter.png` ·
`fig_ablation.png` · `fig_budget.png` (2 subplots)
<!-- fig_landscape.png + fig_regime.png are no longer standalone: they are the (a)/(b) source
     panels for fig_policy_regime (see the combine table above). -->


Palette: muted editorial (steel blue / crimson / sea green / amber / grey), CVD-safe.
Every per-family / per-model chart covers ALL families/models.
