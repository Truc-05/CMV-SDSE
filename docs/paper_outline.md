# Paper outline — *When More Verifiers Are Not More Information: Budget-Aware Fusion of Correlated LLM Verifiers*

Target venue: **Information Fusion (Q1)**. Six top-level sections. The whole paper follows one spine:
*the question is not how many verifiers, but how much new decision-relevant information each dependent
source actually adds — and whether that information is worth acquiring under current cost and risk.*

Narrative beats: ① more verifiers ≠ more information → ② diversity ≠ value → ③ measure conditional
marginal value → ④ fuse adaptively under cost (SDSE) → ⑤ no universally optimal fusion rule → ⑥ the
effects survive broad frozen validation.

---

## Abstract
Paradox → signature finding (N=50 but n_eff≈1, non-universal wording) → count≠diversity≠value →
conditional marginal verification value → SDSE controller → no universal fusion rule → broad frozen
confirmatory validation. Close on the reframed question the editor should remember.

## 1. Introduction  *(beats ①②)*
- 1.1 The paradox: self-consistency / debate / judge panels / cascades all assume *more calls ⇒ more
  evidence ⇒ more reliable*. Teaser the signature finding: **N=50 but n_eff≈1** ("across the evaluated
  task families, repeated same-model verification collapsed to roughly one effective independent source").
- 1.2 The intellectual gap in one line: **source count ≠ statistical diversity ≠ decision value**.
- 1.3 The reframed question (bold, one sentence) + **Contributions C1–C4**.
- **Figure 1 (hero):** the entire paper as one vertical story
  (50 calls → ≈1 effective source → low correlation ≠ useful information → conditional marginal value →
  SCALE | DIVERSIFY | STOP | ESCALATE → fusion rule changes with cost regime).

## 2. Related Work
LLM-as-a-judge quality · self-consistency · multi-agent debate · ensembles / juries · verifier
cascades · cost-aware routing · theory of correlated-verifier cascades. **The missing piece:** how to
fuse multiple *dependent* verifier outputs when dependence, complementary error detection, false
alarms, and compute cost all matter *simultaneously* — and why the three quantities above are usually
conflated.

## 3. Measurement Framework, Setup, and Motivating Observations  *(beats ③④ — the narrative heart)*
*(Merges the framework/estimator, the experimental setup, and the two motivating empirical findings so
the reader arrives at the need for a new metric before any method is proposed.)*
- 3.1 **Framework.** Pool `V`, per-item false-accept `a_iv` (conditional independence, de Finetti);
  reliability `R(S)`, cost `C(S)`, objective `U_λ = R − λC`.
- 3.2 **Dependence estimator.** Beta-Binomial / ceiling-mixture MLE → `ρ_v`, the **effective number of
  verifiers `n_eff(K)`**, blind-spot mass `π₀`.
- 3.3 **Experimental setup.** Verifiers: 5 checkpoints / 3 pretraining families (Qwen, Llama, Mistral)
  / 2 size tiers (3B–14B) — stated scope. Pools = dependence ladder (same_model → same_family →
  cross_family). Task families grouped by *error-generation mechanism*: MAST FC3 (real MAS errors),
  ARC/MMLU/CSQA/TruthfulQA (knowledge/reasoning MCQA), GSM8K (multi-step math), MBPP (code /
  test-grounded). Protocol: k=50 gates/item, item-level majority-reject verdict, error/clean split,
  context modes. Metrics: catch (TPR), false-alarm (FPR), n_eff, marginal catch, utility. **Freeze
  protocol:** controller + hyperparameters frozen before the large-sample confirmatory phase; no
  post-hoc retuning.
- 3.4 **Observation 1 — more sources ≠ more information.** Same-model verification: high `ρ_v`, low
  `n_eff` (≈1); holds across task families; context-invariant (`β_context ≈ 0`). *Source count
  substantially overstates effective information when verifier errors are dependent.*
  Assets: `fig8`, `redundancy_invariant_7family.csv`, `context_decomposition.csv`.
- 3.5 **Observation 2 — diversity ≠ value (the twist).** A near-constant weak verifier can have low
  correlation, *inflate* n_eff, and yet catch **zero** additional errors (dead-verifier
  counterexample). *An independent opinion is useless if it contributes no useful information* — so
  n_eff is not the final metric; this paper exposes its limitation.
  Assets: same-family dead-verifier case, greedy marginal trace.

## 4. Method: Conditional Marginal Value and Adaptive Fusion  *(beat ⑤ — SDSE as a consequence)*
- 4.1 **Conditional marginal verification value.** `Δcatch_i(S) = P(i is caught | S missed it)`; fold
  in false-alarm and compute: `V_i(S) = w_catch·Δcatch_i − w_fa·Δfa_i − λ·C_i`. The shift from
  statistical dependence to *decision-relevant* information fusion.
- 4.2 **The SDSE controller.** Greedy on `V` yields four actions — **SCALE** (repeat while it still
  helps), **DIVERSIFY** (add complementary evidence), **STOP** (marginal value ≤ cost), **ESCALATE**
  (the operating regime needs a source/topology the current cascade cannot represent). Introduce the
  acronym *here*, not up front.
- 4.3 **Guarantees.** Submodularity ⇒ `(1−1/e)` greedy; optimal stopping (non-increasing marginal
  utility); distribution-free probe-driven near-optimality; blind-spot ceiling `R(S) ≤ 1−π₀` — only
  ESCALATE crosses it. (Full proofs in appendix.)

## 5. Experiments & Results  *(beat ⑤ proven + ⑥ validation)*
- 5.1 **Conditional marginal value selects the right sources (controller invariant).** No fixed
  model/family label decides; add ⇔ Wilson CI-low(Δcatch) > 0; the dead verifier is never added; the
  solo/diversify roles flip with the task (e.g. Mistral is the strong solo on code).
  Assets: `{mcqa,gsm8k,mbpp}_marginal.csv`.
- 5.2 **No universally optimal fusion rule (the core IF result).** Scoring rules by `U = catch − ρ·fa`,
  the optimum flips **OR → learned fusion → single-best → majority → AND** as the false-alarm/miss cost
  grows; the controller (run at each ρ) tracks the cost-optimal rule at 4/6 ratios with **one knob**,
  shrinking its set as false alarms get costly. Intuition panel: same-model repeat **+0.003** catch vs
  diverse source **+0.079** (~26× marginal gain). The one regime it cannot track (`ρ≳4`, where
  conjunctive unanimity wins) is the disjunctive→conjunctive boundary named ESCALATE.
  Assets: `fusion_cost_sweep.csv`, `fusion_scale_panel.csv`, `fusion_baselines.csv` → **1 figure + 1
  table in main text**.
- 5.3 **Cross-task & cross-model confirmatory validation.** Frozen controller, large-n (n=150/family)
  across seven families spanning distinct error-generation mechanisms; same-source redundancy persists
  (`n_eff@50 ∈ [1.06,1.21]`), useful diversification is significant per-family (Wilson CI-low > 0), and
  **no post-hoc retuning** was performed. *The same information-fusion phenomenon persists across
  substantially different error mechanisms and task structures.*
  Assets: `redundancy_invariant_7family.csv` → cross-task forest plot.
- 5.4 *(optional)* **Cascade Auditor** — turns the framework into a Scale/Diversify/Stop/Escalate
  recommendation from a cheap probe.

## 6. Discussion & Conclusion  *(elevate + secondary blind-spot + scope)*
- 6.1 **The general information-fusion message.** *How much new decision-relevant information does
  another dependent source contribute — and when is that information worth acquiring?*
- 6.2 **Operating-regime-dependent fusion topology.** The preferred way to *combine* sources adapts to
  utility/cost, not just which sources to pick — this is where ESCALATE (disjunctive → conjunctive /
  out-of-pool) is a real boundary, not a failure.
- 6.3 **Blind spot as a secondary scientific result.** The hypothesized non-zero shared-failure atom
  was **not statistically identified**; we report upper bounds. Power analysis suggests an
  *identifiability* limit (the atom is confounded with a heavy correlated-error tail), not merely a
  sample-size limit. Redundancy/correlation is established strongly; a discrete atom is not — a
  maturity signal, not a negative.
- 6.4 **Limitations.** Verifier scope (open, ≤14B, three families) as stated scope; the disjunctive
  cascade / conjunctive-regime boundary as a deliberate ESCALATE hand-off (learned fusion head is
  future work); conditional-independence assumption; open online/adaptive regret.
- 6.5 **Conclusion.** Close on the reframed question: self-verification does not scale by repetition;
  diversity recovers effective evidence only when it adds decision-relevant information; and the right
  way to fuse changes with cost.
- **Appendices.** Proofs (six theorems) · full per-family tables · π₀ power / identifiability analysis
  · Cascade Auditor details · reproducibility (frozen configs, freeze date).

---

## Four contributions (large and memorable — not 9 small ones)
- **C1 — Dependence-aware measurement.** Verifier count can substantially overstate the effective
  amount of independent evidence; we characterize this redundancy empirically.
- **C2 — Decision value beyond diversity.** Statistical diversity is insufficient for verifier
  selection; we introduce *conditional marginal verification value*, jointly with false-alarm and
  compute costs.
- **C3 — Adaptive budget-aware fusion.** An SDSE controller (SCALE / DIVERSIFY / STOP / ESCALATE); the
  preferred fusion behavior changes across operating regimes rather than any universal fixed rule.
- **C4 — Broad confirmatory validation.** Across heterogeneous task families and multiple model
  families/checkpoints, including a frozen large-sample confirmatory phase: persistent same-source
  redundancy and statistically supported gains from useful diversification.
- *(Blind spot is a **secondary analysis** in §6, not one of the four.)*

## Main-text figure/table budget (~7)
Fig 1 hero story · Fig 2 n_eff dependence ladder (3.4) · Fig 3 dead-verifier counterexample (3.5) ·
Fig 4 SDSE decision schematic (4.2) · Fig 5 fusion cost-sweep, flagship (5.2) · Fig 6 cross-task n_eff
forest (5.3) · Table 1 SCALE vs DIVERSIFY marginal gain · Table 2 seven-family redundancy + invariant.
