# When More Verifiers Are Not More Information: Budget-Aware Fusion of Correlated LLM Verifiers

Working draft — target: **Information Fusion** (Q1). Numbers are frozen
(`docs/METHODOLOGY_FREEZE.md`, 2026-08-09) from `results/tables/` (`main_results_3mode`,
`redundancy_invariant_7family`, `fusion_{baselines,scale_panel,cost_sweep}`, `context_decomposition`,
`cascade_design`). One spine runs through the whole paper: **the question is not how many verifiers,
but how much new decision-relevant information each dependent source actually adds — and whether that
information is worth acquiring under current cost and risk.** Six narrative beats: ① more verifiers ≠
more information → ② diversity ≠ value → ③ measure conditional marginal value → ④ fuse adaptively
under cost (SDSE) → ⑤ no universally optimal fusion rule → ⑥ the effects survive broad frozen
validation. Citation to reconcile before submission: the correlated-cascade *theory* is Han
(arXiv:2607.13918) — the repo earlier mislabelled it "Aksu 2026b."

---

## Abstract
LLM systems increasingly stack verifiers — self-consistency, multi-agent debate, judge panels,
repeated verification, cascades — under the intuition that *more verifier calls ⇒ more evidence ⇒ a
more reliable decision*. That intuition holds only when the added calls carry **new** information. We
show, on real multi-agent traces and across seven evaluated task families, that this assumption
routinely fails: repeated same-model verification is so correlated that **N=50 calls carry the
evidence of roughly one effective independent verifier** (`n_eff@50 ∈ [1.06, 1.21]`), so scaling
self-verification is largely wasted compute. But statistical diversity is *not* the fix either — a
low-correlation verifier can inflate the effective count and still catch zero additional errors. We
therefore recast verifier fusion around **conditional marginal verification value**: how many errors a
source rejects that the current set misses, scored jointly against false-alarm and compute cost. This
yields a budget-aware controller with four actions — **SCALE, DIVERSIFY, STOP, ESCALATE** — and a
guarantee that greedy selection is `(1−1/e)`-optimal. Benchmarked against the standard fusion rules
(OR/AND/majority/Dawid-Skene/weighted-vote/learned stack) on identical verdicts, **no fusion rule is
universal**: the cost-optimal rule flips `OR → learned → single-best → majority → AND` as the
false-alarm/miss cost grows, and the controller tracks it with one knob. All effects survive a frozen,
large-sample confirmatory phase across knowledge, reasoning, math, and code tasks with no post-hoc
retuning. A secondary analysis finds that a discrete *shared blind-spot atom* is **not statistically
identified** (upper bounds only) — redundancy is established strongly; a discrete atom is not.

## 1 Introduction  *(beats ①②)*
LLM agent harnesses increasingly *verify* their own outputs — calling a verifier several times, running
judge panels, debating, or chaining a cascade that accepts only on unanimous pass. Treating each
verifier call as a **decision source**, this is information fusion of *correlated* sources, and the
correlation is the whole story. The field's working intuition is

> more verifier calls ⇒ more evidence ⇒ more reliable decision,

which is sound **only when each call supplies new information**. If 50 verifier calls all repeat the
same blind pattern, then on any information-fusion accounting we do not actually have 50 sources. Our
signature finding, stated without over-generalising:

> **Across the evaluated task families, repeated same-model verification consistently collapsed to
> roughly one effective independent source (`N=50`, `n_eff ≈ 1`).**

This exposes an intellectual gap the literature routinely conflates: **source count ≠ statistical
diversity ≠ decision value**. Number of sources overstates information under dependence; and — as we
show — even genuine statistical diversity does not guarantee decision value. The question we ask, and
the one we want the reader to keep, is:

> **How much new decision-relevant information does another dependent verifier actually contribute —
> and when is that information worth acquiring under current cost and risk?**

**Contributions.**
- **C1 — Dependence-aware measurement.** We empirically characterize information redundancy in LLM
  verification and show that nominal verifier count can substantially overstate the effective amount
  of independent evidence (`n_eff@50 ≈ 1` for same-model cascades across seven families).
- **C2 — Decision value beyond diversity.** We establish that statistical diversity is *insufficient*
  for verifier selection, and introduce **conditional marginal verification value**, which measures
  complementary error detection jointly with false-alarm and computational cost.
- **C3 — Adaptive budget-aware fusion.** We develop an **SDSE** controller that dynamically chooses
  among **SCALE, DIVERSIFY, STOP, ESCALATE**, and show that the preferred fusion behavior changes
  across operating regimes rather than being captured by any universal fixed rule.
- **C4 — Broad confirmatory validation.** We validate the framework across heterogeneous task
  families and multiple model families/checkpoints, including a **frozen** large-sample confirmatory
  phase, showing persistent same-source redundancy and statistically supported gains from useful
  diversification.

*(A hypothesized non-zero shared-failure atom is treated as a secondary analysis, §6.3, not a main
contribution.)*

**Figure 1 (hero).** The whole paper as one vertical story: *50 calls → ≈1 effective source →
low correlation ≠ useful information → conditional marginal value → SCALE | DIVERSIFY | STOP |
ESCALATE → fusion rule changes with cost regime.*

## 2 Related Work
Correlated-vote aggregation is classical for *parallel* voting — Ladha (1992, 1995), Berg (1993),
Boland (1989) show majority-rule effectiveness falls as vote correlation rises; recent LLM work
("Consensus is Not Verification", arXiv:2603.06612) shows parallel agreement cannot scale truthfulness
under correlated errors. Design-effect / ICC has entered LLM evaluation (DiagnosticIQ,
arXiv:2605.08614), and Beta-Binomial overdispersion is standard for judge agreement. The *serial*
correlated-cascade theory we build on is Han (arXiv:2607.13918), which is synthetic-only and supplies
the estimator we reuse (Beta-Binomial / NPMLE, `ρ_v = 1/(a+b+1)`). Adjacent lines — LLM-as-a-judge
quality, self-consistency, multi-agent debate, verifier ensembles/juries, and cost-aware routing —
each optimize one axis in isolation. **The missing piece, which this paper targets, is how to fuse
multiple *dependent* verifier outputs when dependence, complementary error detection, false alarms,
and compute cost all matter simultaneously** — and, in particular, that *number of sources*,
*statistical diversity*, and *decision value* are three different quantities. Our contribution is the
budgeted-fusion framework, the conditional-marginal-value metric, the controller with guarantees, the
fusion-rule ablation, and the first real-MAS + multi-family evidence — not the base estimator.

## 3 Measurement Framework, Setup, and Motivating Observations  *(beats ③④ — the narrative heart)*

### 3.1 Framework
Pool `V` of verifiers, per-call cost `c_v`. On error item `i`, verifier `v` false-accepts with per-item
probability `a_{iv}`, conditionally independent given the item (de Finetti). A cascade `S ⊆ V` catches
`i` iff some `v ∈ S` rejects it:

```
reliability   R(S) = E_i[ 1 − ∏_{v∈S} a_{iv} ]
cost          C(S) = Σ_{v∈S} c_v
objective     max  U_λ = R − λC
```

An **OBSERVE** gate precedes fusion (is the failure evidence inside the verifier's context at all? else
FIX-CONTEXT); the four fusion actions **SCALE / DIVERSIFY / STOP / ESCALATE** are developed in §4 once
the need for them is established.

### 3.2 Dependence estimator
Per-item sufficient statistic `(m_i, K_i)` = unwanted-verdict count over decided gates; fit the nested
Binomial / Beta-Binomial / ceiling-mixture by MLE (`theory/betabinom_mixture.py`). Read off the
intra-item correlation `ρ_v = 1/(a+b+1)`, the design effect, the **effective number of verifiers**
`n_eff(K) = K / (1 + (K−1)ρ_v)`, and the blind-spot mass `π₀`. Indeterminate gates are excluded and
reported.

### 3.3 Experimental setup
**Verifiers.** Five open checkpoints across **three pretraining families and two size tiers**:
Qwen2.5-7B/14B, Llama-3.1-8B / Llama-3.2-3B, Mistral-7B (served via Ollama). This is a stated scope —
enough distinct families that the effects are not a single-architecture artifact, but all open and
≤14B (§6.4). **Pools = a dependence ladder:** `same_model` (repeat one model), `same_family` (two same-
family models), `cross_family` (distinct families). **Task families, grouped by error-generation
mechanism:** MAST FC3 traces (Cemri et al., arXiv:2503.13657) as real multi-agent-system errors;
ARC / MMLU / CSQA / TruthfulQA as knowledge/reasoning MCQA; GSM8K as multi-step mathematical reasoning;
MBPP as program-generation / test-grounded errors. **Protocol:** `k=50` gates/item, item-level
**majority-reject** verdict (`thr=0.5`), error vs clean items, `truncate` and evidence-preserving
context modes. **Metrics:** catch (TPR on errors), false-alarm (FPR on clean), `n_eff`, conditional
marginal catch, utility `U`. **Freeze protocol (methodological anchor).** *The controller and all
hyperparameters were frozen before the large-sample confirmatory phase; no post-hoc retuning was
performed.*

### 3.4 Observation 1 — more sources ≠ more information
Same-model cascades are strongly correlated and their effective count is near one, not `k`. On the
real MAST FC3 traces:

| mode | pool | n | miss | ρ_v | ρ_v CI | **n_eff@50** | blind-spot |
|---|---|---|---|---|---|---|---|
| FM-3.1 | same_model | 40 | 0.91 | 0.54 | [0.36,0.73] | **1.82** | 0.63 |
| FM-3.1 | cross_family | 40 | 0.85 | 0.42 | [0.29,0.57] | **2.35** | 0.39 |
| FM-3.2 | same_model | 40 | 0.99 | 0.52 | [0.17,0.92] | **1.89** | 0.95 |
| FM-3.2 | cross_family | 40 | 0.87 | 0.17 | [0.12,0.27] | **5.24** | 0.00 |
| FM-3.3 | same_model | 40 | 0.82 | 0.82 | [0.69,0.91] | **1.21** | 0.00 |
| FM-3.3 | cross_family | 40 | 0.56 | 0.20 | [0.14,0.28] | **4.62** | 0.00 |

**Fig. 1a: 50 calls ≠ 50 independent verifiers** — same-model 1.2–1.9, never 50; the same-model null
tightens to `n_eff@50 ∈ [1.06, 1.21]` at confirmatory scale across all seven families (Fig. 7, §5.3). The
effect is **intrinsic, not a context artifact**: holding seeds fixed so context is the only variable,
an evidence-preserving context mode leaves `n_eff` and blindness essentially unchanged
(`context_decomposition.csv`, `β_context ≈ 0`):

| mode | n_eff (trunc→evid) | blind (trunc→evid) | β_context |
|---|---|---|---|
| FM-3.1 | 1.82 → 1.75 | 0.63 → 0.72 | −0.09 |
| FM-3.2 | 1.89 → 1.89 | 0.95 → 0.95 | +0.00 |

*Source count substantially overstates effective information when verifier errors are dependent.* High
miss rates (0.82–0.99) are themselves a finding: these small verifiers are weak on FC3.

### 3.5 Observation 2 — diversity ≠ value (the twist)
If low correlation were the goal, one could add any low-agreement verifier and raise `n_eff`. But a
near-constant weak verifier can be **statistically diverse yet decision-useless**: the 3B `llama32_3b`
verifier has low correlation with the rest, *raises the apparent effective count*, and yet catches
**zero** errors the current set misses — so a controller must, and does, never add it (§5.1).

> **An independent opinion is useless if it contributes no useful information.**

This is why we do not sell `n_eff` as the final metric: the paper itself exposes its limitation. Both
observations point to the same missing quantity — not *how many* sources, nor *how uncorrelated*, but
*how much new decision-relevant information a source adds given what is already known*.

## 4 Method: Conditional Marginal Value and Adaptive Fusion  *(beat ⑤ — SDSE as a consequence)*

### 4.1 Conditional marginal verification value
Instead of asking how accurate a verifier is, or how different it is from the others, we ask what it
adds given the current set `S`:

```
Δcatch_i(S) = P( source i catches an error | S missed it )
V_i(S)      = w_catch · Δcatch_i(S)  −  w_fa · Δfa_i(S)  −  λ · C_i
```

`Δcatch` is the only in-pool source of new evidence; folding in the false-alarm increment `Δfa` and
the compute cost `C_i` turns statistical dependence into a single **decision-relevant** fusion score.
The **2-verdict probe** (`theory/probe.py`) estimates these marginal gains directly and unbiasedly;
`R=2` joint rounds already recover the optimal cascade order.

### 4.2 The SDSE controller
Greedy maximization of `V` yields exactly four actions — the controller does not need to be posited
separately, it *is* the argmax of marginal value under cost:

| action | meaning | trigger |
|---|---|---|
| **SCALE** | repeat the current source | its self-`n_eff` has not yet saturated |
| **DIVERSIFY** | add a complementary source | `Δcatch` significant (Wilson CI-low > 0) |
| **STOP** | accept | marginal utility `u_t = Δ_t − λc_t ≤ 0` |
| **ESCALATE** | go out-of-pool / stronger / switch fusion topology | the regime needs evidence the cascade cannot represent (§5.2, §6.2) |

### 4.3 Guarantees *(proofs in `docs/theory_cascade_design.md`)*
- **Thm 1.** `R` is monotone submodular ⇒ greedy on marginal value is `(1−1/e)`-optimal; marginal
  value = expected catch on *survivors* ⇒ **Cor. 1: value ≠ accuracy**.
- **Thm 2.** For every in-pool cascade `R(S) ≤ 1−π₀`; only enlarging `V` (ESCALATE) crosses the ceiling.
- **Thm 3.** Probe-driven greedy is distribution-free and margin-free:
  `R(S_k) ≥ (1−1/e)·OPT_k − 2k·ε(R)`, `ε(R)=√(log(2mk/δ)/(2nR))`; `R=2` recovers the optimal order.
- **Thm 4.** Marginal utility `u_t` is non-increasing ⇒ stop at first `u_t ≤ 0` is optimal on the path.
- **Thm 5.** Safe stopping: stop only when the UCB of the next marginal is below cost; no premature
  stop w.p. `1−δ`.
- **Thm 6.** If verifiers are nested / stochastically ordered, accuracy order = optimal; the selection
  gap scales with a complementarity functional `χ` (`χ=0` for same-model ⇒ the only lever is STOP).

## 5 Experiments & Results  *(beat ⑤ proven + ⑥ validation)*

### 5.1 Conditional marginal value selects the right sources
The controller (`theory/marginal_value.py`) adds a candidate **only when its Wilson CI-low on `Δcatch`
> 0**, and no fixed model/family label decides the outcome — the *conditional marginal catch on this
task* does. The dead 3B verifier `llama32_3b` (zero unique catch) is **correctly never diversified in**
on any family; conversely a verifier that is dead at small-n can become significant at scale and is
then added. The solo/diversify **roles flip with the task**: e.g. on code (MBPP) Mistral is the
strongest *solo* verifier while Llama/Qwen diversify, whereas on MCQA Qwen is the solo and Mistral
diversifies. *No label is destiny; only measured marginal value is.* (`{mcqa,gsm8k,mbpp}_marginal.csv`.)

### 5.2 No universally optimal fusion rule *(the core information-fusion result)*
We benchmark the controller against the fusion rules a practitioner would reach for, on the *same*
per-item verdicts, over the six identical-setup confirmatory families (MCQA×4 + GSM8K + MBPP; MAST-3.3
excluded — its verifier×item matrix is not rectangular). Two results.

*(i) Repetition is worthless, diversity is not — in fusion terms.* Adding an independent same-model
repeat lifts error-catch by **+0.003** (macro-avg); adding a diverse source lifts it by **+0.079**
(~26×). This is `n_eff@50 ≈ 1` restated as a fusion result (`fusion_scale_panel.csv`).

*(ii) No universal fusion rule.* Scoring each rule by utility `U = catch − ρ·fa` (`ρ` = false-alarm/
miss cost), the optimum flips as `ρ` grows; run *at* each `ρ`, the frozen controller (`w_fa=ρ`,
OR-cascade of its selected set) tracks the cost-optimal fixed rule at **4/6 cost ratios with one knob**,
shrinking its set **2.67 → 1.0** verifiers as false alarms get costly (`fusion_cost_sweep.csv`):

| `ρ` = fa/miss | 0.0 | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|
| best fixed rule | OR | learned stack | learned stack | single-best | MAJORITY | AND |
| best-fixed `U` | 0.954 | 0.761 | 0.644 | 0.475 | 0.275 | 0.093 |
| **controller `U`** | 0.953 | 0.761 | 0.588 | 0.460 | 0.259 | −0.034 |
| controller set-size | 2.67 | 2.33 | 2.17 | 1.50 | 1.00 | 1.00 |

The claim is **not** that OR-cascade is the best fusion rule — at `ρ=1` naive OR of all four verifiers
is *dominated* by a selective rule; the claim is that the controller selects the cost-appropriate
operating point. The one regime it cannot track (`ρ ≳ 4`) is a **deliberate scope boundary**: the
reject-any cascade is *disjunctive* and only adds false alarms, whereas the precision-critical optimum
is *conjunctive* (unanimity) — a topology outside the cascade family. This is exactly where the
framework prescribes **ESCALATE** (switch fusion mode / go out-of-pool) rather than adding gates.

### 5.3 Cross-task and cross-model confirmatory validation
With the controller frozen, we run the confirmatory phase at `n=150` error items/family (150×50 gates ×
3 pools = 11,250 backend calls/family, 0 unparsed, 0 context overflow) across seven families spanning
distinct error-generation mechanisms. Same-source redundancy persists — `n_eff@50 ∈ [1.06, 1.21]` — and
useful diversification is significant per family (Wilson CI-low > 0; pooled 71/120 residual catch)
(`redundancy_invariant_7family.csv`):

| family | n | same-model ρ_v | **n_eff@50** | solo → diversified-in | dead never-added |
|---|---|---|---|---|---|
| MAST-3.3 | 40 | 0.821 | **1.21** | — | — |
| ARC | 150 | 0.901 | **1.11** | qwen7b → mistral7b (11/15) | llama32_3b |
| MMLU | 150 | 0.881 | **1.13** | qwen7b → mistral7b (21/26) | llama32_3b |
| CSQA | 150 | 0.900 | **1.11** | qwen7b → mistral7b (9/16) | llama32_3b |
| TruthfulQA | 150 | 0.915 | **1.09** | llama31 → mistral7b (19/36) | llama32_3b |
| **GSM8K** (reasoning) | 150 | 0.856 | **1.16** | llama31 → mistral7b (3/7) | llama32_3b |
| **MBPP** (code) | 150 | 0.938 | **1.06** | mistral7b → llama31 (8/20) | llama32_3b |

*The same information-fusion phenomenon persists across substantially different error mechanisms and
task structures*, and — because the controller was frozen first — with **no post-hoc retuning**.

### 5.4 The Cascade Auditor *(optional)*
`theory/auditor.py` ingests a pool (or a small probe) and returns per-verifier self-`n_eff` (SCALE
redundancy), the pool ordering, `k*`, reliability, the blind-spot ceiling, **Verification Debt**
`= calls − n_eff`, **Efficiency** `= n_eff/calls`, and a Scale/Diversify/Stop/Escalate recommendation
with a warning ("adding N more calls is statistically useless"). On the real pool it recommends
ESCALATE when the target exceeds the ceiling and STOP when it is met.

## 6 Discussion & Conclusion

### 6.1 The general information-fusion message
The unifying question is not *how many* verifiers or *how diverse*, but **how much new decision-relevant
information another dependent source contributes — and when that information is worth acquiring under
cost and risk.** Source count overstates information under dependence (§3.4); statistical diversity
does not imply value (§3.5); conditional marginal value makes the trade-off explicit (§4); and the
right amount to acquire is set by marginal utility, not by a fixed depth or rule (§5).

### 6.2 Operating-regime-dependent fusion topology
The preferred way to *combine* sources adapts to utility/cost, not merely which sources to pick: the
optimal rule flips `OR → learned → single-best → majority → AND` with the false-alarm/miss cost (§5.2).
The disjunctive reject-any cascade is optimal only in the miss-dominated regime; the precision-critical
regime needs conjunctive unanimity — a genuinely different topology, which is why **ESCALATE** is a real
boundary in the framework rather than a failure mode. A learned conjunctive fusion head is future work.

### 6.3 Blind spot — a secondary scientific result
We hypothesized that heavily correlated verification could produce a non-zero *shared-failure atom*
(a mass of items no in-pool cascade ever catches). On the large-n data this atom is **not statistically
identified** on any family: the profile CI on `π₀` touches 0 and the boundary-LRT is non-significant
everywhere (upper bounds `π₀ ≤ 0.13–0.40`). A pre-registered power analysis indicates an
**identifiability** limit, not a sample-size limit — the atom is confounded with a heavy
correlated-error tail (`ρ_v ≈ 0.71–0.85`), and TruthfulQA stays under 0.8 power even at `n=2000`.

> *Blind-spot mass was not statistically identified; we report upper bounds rather than claim the
> existence of a non-zero shared-failure atom. A high apparent blind-spot mass does not imply the atom
> is identifiable, because persistent failures can be confounded with a heavy correlated-error tail.*

Redundancy/correlation is established strongly; a discrete atom is not — a maturity signal, and the
reason blind spot is a secondary analysis, not a main claim.

### 6.4 Limitations
1. **Verifier scope (stated, not a defect).** Five checkpoints / three families (Qwen, Llama, Mistral)
   / two size tiers (3B–14B) — enough that the effect is not single-architecture, but all open and
   ≤14B; we do not claim frontier/closed models or other families (Gemma/Phi/…) behave identically.
2. **Disjunctive scope (deliberate = ESCALATE).** The reject-any cascade cannot represent the
   conjunctive optimum of the precision-critical regime (§5.2, §6.2); a learned fusion head is future work.
3. `n_eff` assumes conditional independence given the item (supported: verdict autocorrelation ≈ 0,
   product formula predicts held-out outcomes to ≈ 0.03) but not proven for all pools.
4. The ordering (selection) gain is small on real pools (Thm 6); the value is the framework, the
   guarantees, and the SCALE/DIVERSIFY/STOP/ESCALATE decision — not an accuracy jump.
5. Thm 5's online/adaptive *regret* (no separate probe budget) is open.

### 6.5 Conclusion
Self-verification does not scale by repetition: on real multi-agent traces and across seven task
families, 50 same-model verifier calls are worth ~1 independent one, and the redundancy is intrinsic,
not a context artifact. Diversity recovers effective evidence **only when it adds decision-relevant
information**, and the right way to fuse changes with cost. Measuring conditional marginal value — and
acting on it with SCALE / DIVERSIFY / STOP / ESCALATE — turns "how many verifiers" into the question
that matters: *how much new information is the next source actually worth?*

## References (verify before submission)
Han 2026 (arXiv:2607.13918); Odds Law (arXiv:2606.15712); Cemri et al. 2025 MAST (arXiv:2503.13657,
NeurIPS D&B); Consensus is Not Verification (arXiv:2603.06612); DiagnosticIQ (arXiv:2605.08614);
Ladha 1992/1995; Berg 1993; Boland 1989; Nemhauser–Wolsey–Fisher 1978; Chernoff 1954 / Self–Liang 1987.

## Figures & tables
**Fig 1** hero story (the six-beat spine) · **Fig 2** effective-verifiers / dependence ladder (§3.4,
`fig8`) · **Fig 3** dead-verifier counterexample (§3.5) · **Fig 4** SDSE decision map (§4.2, `fig7`
flagship) · **Fig 5** fusion cost-sweep, controller vs fixed rules (§5.2, flagship IF result) ·
**Fig 6** cross-task `n_eff` forest plot (§5.3). Tables: **T1** SCALE vs DIVERSIFY marginal gain
(`fusion_scale_panel`); **T2** seven-family redundancy + invariant (`redundancy_invariant_7family`);
supporting `main_results_3mode`, `context_decomposition`, `fusion_cost_sweep`, `cascade_design`.
