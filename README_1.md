# Budget-Aware Fusion of Correlated LLM Verifiers
### *When to Scale, Diversify, Stop, or Escalate*

> **The question.** *How should multiple **dependent** verifiers actually be fused under a budget?*
> A serial verifier cascade — accept an answer only if `k` verifier calls all pass — is **information
> fusion of correlated decision sources**, and the correlation is the whole story: on real traces,
> **50 same-model verifier calls carry the evidence of only ~1 independent verifier**, so scaling
> self-verification is wasted compute. We measure a verifier by its **conditional marginal catch**
> (errors it rejects that the current set misses), and from that + false-alarm cost + compute cost a
> frozen controller decides **SCALE / DIVERSIFY / STOP / ESCALATE**.

> Target venue: **Information Fusion (Q1)**. Draft: `docs/paper_draft.md`. Numbers are **frozen**
> (`docs/METHODOLOGY_FREEZE.md`, 2026-08-09); controller + hyperparameters locked. **Confirmatory
> large-n DONE:** 4 MCQA families (2026-08-10) **+ GSM8K reasoning + MBPP code (2026-08-11)** at n=150
> err/family (150×50 gates × 3 pools), no re-tuning — redundancy null held across **seven families**
> (`n_eff@50 ∈ [1.06,1.21]`), diversify signal now **powered** (CI-low>0 on all six greedy families).
> The secondary blind-spot (`π₀`) claim is **CLOSED as a bounded negative result** (C11): not
> statistically identified, upper bounds only — it does not gate the paper.

---

## 0. The seven contributions (paper spine)

The paper's axis is **budget-aware fusion of dependent verifiers**, not "validating a blind-spot /
concavity theory". Seven contributions, in narrative order:

1. **50 verifiers ≠ 50 independent sources.** Same-model self-verification saturates —
   `n_eff@50 ≈ 1.06–1.21` across seven confirmatory benchmarks (MCQA×4 + GSM8K reasoning + MBPP code).
   Repetition (SCALE) buys almost nothing.
2. **Statistical diversity ≠ decision value.** A *low-correlation* verifier can still be completely
   useless; a dependence metric **alone** can pick the wrong source. Value is not correlation.
3. **Conditional marginal value.** Measure a verifier directly by how many errors it rejects that
   the current verifier set **misses** — the only in-pool source of new evidence.
4. **Budget-aware adaptive fusion.** From marginal value + false-alarm cost + compute cost, decide
   **SCALE / DIVERSIFY / STOP** (and ESCALATE out-of-pool past the ceiling).
5. **Multi-benchmark confirmation.** Controller **frozen before** large-n; the diversification
   signal is still **significant on ARC, MMLU, CSQA, and TruthfulQA** (per-family Wilson CI-low > 0).
6. **Blind spot — secondary negative result.** We do **not** establish a non-zero shared-failure
   atom. *Blind-spot mass was not statistically identified; we report upper bounds rather than claim
   the existence of a non-zero shared-failure atom.* And a methodological note: *a high apparent
   blind-spot mass does not imply the atom is identifiable, because persistent failures can be
   statistically confounded with a heavy correlated-error tail.* This makes the paper more
   disciplined: hypothesis → test → insufficient evidence → no claim, plus *why* more samples need
   not fix it (see §3.9).
7. **The controller tracks the cost-optimal fusion rule.** Benchmarked against the standard rules
   (OR/AND/majority/Dawid-Skene/weighted-vote/pattern-Bayes) on the *same* verdicts, **no fusion rule
   is universal** — the winner flips with the false-alarm/miss cost. The frozen controller follows the
   cost-optimal rule with **one knob** (4/6 cost ratios), shrinking its set as false alarms get
   costly; the single regime it cannot track (`ρ≳4`, where unanimity wins) is the **disjunctive→
   conjunctive** boundary the framework names ESCALATE (see §3.10).

---

## 1. Framework

Pool `V` of verifiers, per-call cost `c_v`. On error item `i`, verifier `v` false-accepts with
per-item probability `a_{iv}` (conditionally independent given the item — de Finetti). A cascade
`S⊆V` catches `i` iff some `v∈S` rejects it.

```
reliability   R(S) = E_i[ 1 − ∏_{v∈S} a_{iv} ]
cost          C(S) = Σ_{v∈S} c_v
objective     max U_λ = R − λC
```

Five operational decisions, chained **OBSERVE → SELECT/DIVERSIFY → SCALE/STOP → ESCALATE**:

| decision | meaning | when |
|---|---|---|
| **OBSERVE** | is the failure evidence even inside the verifier's context? | else FIX-CONTEXT |
| **DIVERSIFY** | add a *complementary* verifier | the only in-pool source of new evidence |
| **SCALE** | repeat the same verifier | useless once self-`n_eff` saturates |
| **STOP** | accept; more verification costs more than it returns | `u_t ≤ 0` |
| **ESCALATE** | go out-of-pool / stronger / human | the only way past the ceiling `1−π₀` |

The Scale/Diversify/Stop/Escalate map is the flagship figure (`fig7`), anchored by real pools.

---

## 2. Contributions & status  *(checklist — [done] = shipped & frozen)*

| | Contribution | Status |
|---|---|---|
| **C1** | **Framework** — budget-aware fusion of correlated verifiers, `max U_λ = R−λC`, 5 decisions | **[done]** |
| **C2** | **Estimator** — item-level Beta-Binomial / ceiling mixture MLE → `ρ_v`, `n_eff(K)`, blind-spot `π₀` | **[done]** |
| **C3** | **Theory** — 6 theorems w/ proofs (`docs/theory_cascade_design.md`) | **[done]** |
| **C4** | **MAST empirics** — 3 FC3 failure modes; same-model `n_eff@50` 1.2–1.9 vs cross-family 2.3–5.2 | **[done]** |
| **C5** | **7-family redundancy replication** — MAST-3.3 + ARC/MMLU/CSQA/TruthfulQA + **GSM8K (reasoning)** + **MBPP (code)**, same-model `n_eff@50 ∈ [1.06, 1.21]` (large-n, n=150) | **[done]** |
| **C6** | **Controller invariant** — no label decides; add ⇔ Wilson CI-low(Δcatch)>0; dead verifier `llama32_3b` never added, holds 4/4 MCQA | **[done]** |
| **C7** | **Context confound locked** — `n_eff` context-invariant, `β_context ≈ 0` | **[done]** |
| **C8** | **Cascade Auditor** — probe → `n_eff`, ceiling, order, `k*`, Verification-Debt, S/D/S/E recommendation | **[done]** |
| **C9** | **Methodology freeze** — controller + hyperparameters locked 2026-08-09 | **[done]** |
| **C10** | **Confirmatory large-n** — frozen pipeline re-run at n=150 err/family; redundancy null held, **diversify signal now powered** (60/93 residual, per-family CI-low>0) | **[done — 2026-08-10]** |
| **C11** | **Blind-spot `π₀` (secondary)** — re-fit on large-n: **0/5 families** reach existence (CI touches 0, atom LRT n.s.). Report **upper bounds** (`π₀ ≤ 0.13–0.40`). Not an n-problem: atom is confounded with the ρ_v≈0.85 Beta-Binomial tail. Will not gate the paper | **[done — bounded, not established]** |
| **C12** | **Fusion-rule ablation** — vs OR/AND/majority/Dawid-Skene/weighted/pattern-Bayes on the same verdicts; no universal rule (winner flips with fa/miss cost), controller **tracks the cost-optimal rule 4/6** with one knob; disjunctive-cascade ceiling → **ESCALATE** to conjunctive fusion at `ρ≳4` | **[done — 2026-08-11]** |

**62 unit tests pass** (`~/venvs/env/bin/python -m pytest -q`).

### Paper-hardening pass — 2026-08-15  *(all in `nd.md`; analyses offline, no GPU)*

**DONE (shipped into the paper + reconcile to `U=0.486`):**
- **Theory scoped** — `(1−1/e)` now explicitly on the catch-max core only + path-optimal stopping (abstract/prop/conclusion); probe **bound** separated from the empirical "R=1–2 recovers order".
- **Discussion + Conclusion + Highlights** written (were empty); Introduction reframed fusion-first.
- **Pareto** `figures/fig_pareto.pdf` (`scripts/pareto_plot.py`) — CMV-SDSE is the **sole non-dominated** policy (0.26 s / U 0.486 vs same-model 30.4 s / 0.233).
- **Held-out CMV** (`§holdout`, `scripts/reviewer_holdout_escalate.py`) — disjoint calib/test, 200 splits: test **U 0.475** vs in-sample 0.495; beats every baseline out-of-sample.
- **ESCALATE validated** (`§escalate_exp`) — in-pool exhausted → escalate; genuine 7B→14B `qwen14b`: same-family barely helps (0.103→0.128), cross-family closes it (→0.641). Escalation value = complementarity, not scale.
- **Cross-task transfer 6×6 + re-probe** (`§transfer`, `scripts/transfer_matrix.py`, `reprobe_transfer.py`) — zero-shot regret **0.130@ρ=0.25 → 0.000 with a 1-round target re-probe** (1/50 of a full re-fit).
- **Fusion-literature baselines** (`§fusionselect`, `scripts/fusion_selection_baselines.py`) — **GES** (Caruana ICML'04) + **mRMR** (Peng TPAMI'05): CMV **matches GES in-sample** (0.486 vs 0.487) and **wins held-out** (0.475 vs 0.464), **dominates mRMR** (0.367). *→ paste 2 BibTeX (`caruana2004ensemble`, `peng2005mrmr`) into `cas-refs.bib`.*
- **2 real bugs fixed** — regret `0.056@ρ=0.5` (was mislabeled ≤0.016); dangling `\Cref{sec:fusion_rules}`.
- **24B escalation plumbing** — `cascade/backend_factory.py` (provider dispatch), `configs/models.yaml` verifier `mistral_small_24b` + pools `same_model_mistral24b`/`escalate_mistral24b`.

**24B robustness [done — 2026-08-16]** — Mistral-Small-24B (open, public 24B → drops the "≤14B" limitation). All 6 families collected (k=3, 225 rec each: 150 err + 75 ok; invisible to the `n_gates==50` frozen tables). Offline analysis `scripts/analyze_escalation.py mistral_small_24b` → `results/tables/rev_escalate_mistral_small_24b.csv`:
  - macro over 6 families, ρ=0.25: **CMV-SDSE→external U=0.780** (catch 0.873, fa 0.373) at **44.7% external calls** > Always-external 0.688 @100% > Random-at-same-rate 0.727 > Local-only 0.654. Selective escalation beats always-external at <half the calls — the qwen14b story reproduced on a public 24B.

**Status 2026-08-16 — freeze briefly re-opened for one run.** Experiments were frozen after the k=3 24B escalate landed; then deliberately (user-authorised) re-opened for the **24B redundancy** run below — which produced a *surprise* (the 24B is **less** redundant, not more), so ≤14B is **kept** and the headline goes 2-tier. Transfer branch stays frozen; frozen controller + `n_gates==50` tables untouched (isolated pool name + `_prep` probe-exclusion).

**Reviewer-critique revisions [`nd.md`, 2026-08-16]** — 24B escalate result written into `§escalate_exp`; **≤14B relaxed** (setup + limitations, escalation now tested above 14B); honesty fixes: Cohen's *d* flagged **family-level (N=6)** in `tab:bootstrap`; "generalizes"→held-out **items** (cross-family weaker); **ρ=4 negative-utility (−0.034 vs best-fixed 0.093) failure surfaced in main text**; Fig 4 ceiling marked *schematic, not fitted*; 2026 preprints labelled `(preprint)`; λ-sensitivity caveat.

**Repo is now git** (was not a repo) — baseline `be291ab` + branch `reviewer-revisions-2026-08-16`; frozen `configs/cascade_thresholds.yaml` **SHA-256 `af8e9dcb6b3b…`** cited in the freeze statement (auditable: proves the config is fixed / silent-change-detectable, *not* the freeze date).

**24B same-model redundancy — SURPRISE FINDING (full parity 7/7 DONE [2026-08-17]).** Repeating the open 24B 50× is *materially less* redundant than the ≤14B pool: n_eff@50 = 1.43 / 1.79 / 1.62 (ARC/GSM8K/MBPP, ρ_v 0.55–0.69) vs ≤14B 1.06–1.21 (ρ_v 0.86–0.94). The 24B's verdicts are genuinely more stochastic across repeats (GSM8K 34% unanimous vs qwen7b 79–93%) — **not a pipeline bug**: 4 sanity checks pass (matched temp/seed=f(item,rep), 0 indeterminate, same `fit_mixture`, and the 24B fitter reproduces the qwen7b band exactly). **Framing discipline: NOT "scale-dependent"** (no controlled size ladder; 24B differs in family/training/tuning) → *"redundancy persists beyond the ≤14B pool but its magnitude is not invariant across checkpoints; we do not attribute the difference to scale alone."* **Core ≤14B band 1.06–1.21 kept** (frozen-confirmed scope); headline → **2-tier** (50 calls ≪ 50 sources [universal] + severity varies across checkpoints [nuance]). Directly answers reviewer #2. **Parity 7/7 DONE [2026-08-17]:** `scripts/run_mistral24b_redundancy_full.sh` — 6 confirmatory families @ n=150 + MAST-3.3 @ n=40, k=50 (isolated pool `redun_mistral24b`). Final `n_eff@50`: ARC 1.44 / MMLU 1.74 / CSQA 1.53 / TruthfulQA 1.58 / GSM8K 1.74 / MBPP 1.54 (ρ_v 0.565–0.690); **MAST-3.3 2.29** (n=40, ρ_v 0.425 — real multi-agent traces, the *least* self-redundant family). Full **band [1.44, 2.29]**, all ≫1 and clear of the ≤14B [1.06,1.21] band — the surprise holds at parity. (The MAST leg first crashed on a real loader bug — `mast_loader` `int(None)` on present-but-null FM keys, fixed `57817e4` — then re-ran clean.) `results/tables/redun_mistral24b_neff.csv`. Not yet written to paper; `fig_sevenfamily` untouched pending the paper pass.

**Done since (all committed):** **#8** — 3 same-family escalation pairs (offline; cross-family beats same-family 7/8, +0.65 vs +0.49); **#9** — held-out utility-stability `P(regret≤ε)` (macro 90% within 0.05; ARC clean dissociation, MBPP honest exception); **reviewer-2 theory pass** — fixed the `F(S)`-submodular error, scoped stopping to the catch–compute objective, weakened independence (R,F coverage of observed verdicts), reliability target as a hard constraint, `V_g(v|S)` acq-vs-fusion generalization, contributions **C1–C5→C1–C4**, ESS wording + restrained tone; **reproducibility fix** (`eb5fb25`) — `_prep` excludes out-of-pool probes so all **10 frozen `_prep`-based scripts reproduce byte-identical** (a re-run had drifted, held-out agree 0.647 vs 0.760).

**Next:** finish the MAST-3.3 leg → lock final 7/7 redundancy → write the 2-tier result + update `fig_sevenfamily`. **#3 gemma [RESULT 2026-08-17]:** the diversity run completed (GPU0, k=3, `same_model_gemma`); the scan's earlier *"NONE found"* was an **artifact** — `reviewer_analyses.load()` hard-filtered `n_gates==50` and silently dropped the k=3 gemma probe (fixed `2c34b1b`, `load(keep_all_k=True)`; scan now tags each verifier's `k`). With gemma visible, **CSQA gemma2_9b is the non-trivial counterexample** C2 wanted: strong (catch 0.66) + diverse (ρ 0.48 vs qwen7b) yet **useless** (0/16 of the solo's residual errors, Δcatch 0.0). **PROBE-ONLY** (k=3 vs solos k=50; 0/16 wide CI) — a **k=50 CSQA confirm is running now** (`redun_gemma` pool, GPU0); if it holds, swap the trivial dead-verifier evidence for this positive counterexample. **Blocker (user):** no `cas-refs.bib` in repo → 16 `\cite` keys compile as `[?]`; verify the 2026 keys `consensus2026verification`/`han2026cascade` are real.

**Evidence stack:** `50→n_eff≈1 (≤14B)` → `diversity≠value` → CMV/CMV-SDSE → held-out + theory + ESCALATE + Pareto → transfer fails selectively → 1-round re-probe recovers native → escalation validated on a public 24B (escalate ≤14B-limit dropped) → redundancy is **checkpoint-dependent** (24B less redundant; ≤14B band retained, headline 2-tier).

### Claim discipline (do NOT overclaim)
- ✅ "**replicated across seven evaluated benchmark/task families**" (MCQA×4 + MAST-3.3 + GSM8K
  reasoning + MBPP code)
- ❌ "works on any benchmark / any model"
- The **redundancy null is the strong, well-powered result** (now **150 items × 50 gates /
  family**, n_eff@50 ∈ [1.06, 1.21]).
- **Diversify signal is now powered (C10):** at n=150 every family's diversify marginal-catch
  Wilson **CI-low > 0** (ARC 0.48 / MMLU 0.62 / CSQA 0.33 / TruthfulQA 0.37 / GSM8K 0.16 / MBPP 0.22),
  pooled **71/120** residual across the six greedy families.
  The earlier "underpowered, directional (9/15)" caveat no longer applies.
- **Blind-spot `π₀` is a secondary NEGATIVE result (CLOSED).** Large-n re-fit did not identify the
  atom on any family (CI touches 0, atom-LRT n.s.); we report **upper bounds**, not existence. Say
  *"blind-spot mass was not statistically identified"* — never *"shared blind spot proven"*.

---

## 3. What is done — detailed

### 3.1 Estimator & framework (C1–C2) **[done]**
Per-item sufficient statistic `(m_i, K_i)` = unwanted-verdict count over decided gates; fit
nested Binomial / Beta-Binomial / ceiling-mixture by MLE (`theory/betabinom_mixture.py`). Read off
`ρ_v = 1/(a+b+1)`, design effect, `n_eff(K)=K/(1+(K−1)ρ_v)`, blind-spot mass `π₀`. Indeterminate
gates are excluded and reported.

### 3.2 Theory — 6 theorems (C3) **[done]**  *(proofs in `docs/theory_cascade_design.md`)*
- **Thm 1** `R` is monotone submodular ⇒ greedy on marginal value is `(1−1/e)`-optimal;
  marginal value = expected catch on *survivors* ⇒ **Cor. 1: value ≠ accuracy**.
- **Thm 2** blind-spot ceiling: `R(S) ≤ 1−π₀` for every in-pool cascade; only ESCALATE crosses it.
- **Thm 3** probe-driven greedy is near-optimal, distribution-free & margin-free:
  `R(S_k) ≥ (1−1/e)·OPT_k − 2k·ε(R)`, `ε(R)=√(log(2mk/δ)/(2nR))`; **`R=2` recovers the optimal order**.
- **Thm 4** marginal utility `u_t = Δ_t − λc_t` is non-increasing ⇒ stop at first `u_t ≤ 0` is optimal.
- **Thm 5** safe stopping via UCB of the next marginal; no premature stop w.p. `1−δ`.
- **Thm 6** if verifiers are nested / stochastically ordered, accuracy order = optimal; the
  selection gap scales with a complementarity functional `χ` (`χ=0` for same-model ⇒ lever is STOP).

### 3.3 MAST empirics — same-model self-verification saturates (C4) **[done]**
MAST FC3 traces (Cemri et al. 2503.13657) as ground-truth errors; small open verifiers via Ollama
— models actually present in the frozen runs: **Qwen2.5-7B, Qwen2.5-14B, Llama-3.1-8B,
Llama-3.2-3B, Mistral-7B** (pools `same_model` / `same_family` / `cross_family_3`); `k=50`,
40 error + 20 clean items/mode.

| mode | pool | miss | ρ_v | **n_eff@50** | blind-spot |
|---|---|---|---|---|---|
| FM-3.1 | same_model | 0.91 | 0.54 | **1.82** | 0.63 |
| FM-3.1 | cross_family | 0.85 | 0.42 | **2.35** | 0.39 |
| FM-3.2 | same_model | 0.99 | 0.52 | **1.89** | 0.95 |
| FM-3.2 | cross_family | 0.87 | 0.17 | **5.24** | **0.00** |
| FM-3.3 | same_model | 0.82 | 0.82 | **1.21** | 0.00 |
| FM-3.3 | cross_family | 0.56 | 0.20 | **4.62** | 0.00 |

Diversity also collapses a persistent blind spot (**FM-3.2: 0.95 → 0.00**). High miss rates
(0.82–0.99) are themselves a finding: these small verifiers are weak on FC3.

### 3.4 Seven-family redundancy replication (C5) **[done — large-n]**
Same estimator, same frozen settings, on 4 non-MAST MC-QA benchmarks + two confirmatory large-n
families spanning **reasoning (GSM8K)** and **code (MBPP)** (each: error item = a *wrong* option /
buggy solution, clean item = the correct one — identical polarity to the MAST loader, so the whole
downstream is reused unchanged). **Confirmatory scale: n=150 error items / family, 3 pooling
conditions × 225 items × 50 gates = 11,250 backend calls/family, 0 unparsed, 0 context overflow.**
Table: `results/tables/redundancy_invariant_7family.csv` (5-family artifact kept byte-for-byte).

| family | n | same-model ρ_v | **n_eff@50** | miss | solo catch | diversified in | dead never-added | invariant |
|---|---|---|---|---|---|---|---|---|
| MAST-3.3 | 40 | 0.821 | **1.21** | 0.823 | — | — | — | — |
| ARC | 150 | 0.901 | **1.11** | 0.118 | qwen7b 0.90 | mistral7b (11/15) | llama32_3b | ✅ |
| MMLU | 150 | 0.881 | **1.13** | 0.178 | qwen7b 0.83 | mistral7b (21/26) | llama32_3b | ✅ |
| CSQA | 150 | 0.900 | **1.11** | 0.118 | qwen7b 0.89 | mistral7b (9/16) | llama32_3b | ✅ |
| TruthfulQA | 150 | 0.915 | **1.09** | 0.448 | llama31 0.76 | mistral7b (19/36) | llama32_3b | ✅ |
| **GSM8K** (reasoning) | 150 | 0.856 | **1.16** | 0.625 | llama31 0.95 | mistral7b (3/7) | llama32_3b | ✅ |
| **MBPP** (code) | 150 | 0.938 | **1.06** | 0.460 | mistral7b 0.87 | llama31 (8/20) | llama32_3b | ✅ |

**same-model `n_eff@50 ∈ [1.06, 1.21]` across seven families** — a very consistent redundancy null,
now well-powered (150×50 gates/family). Diversify marginal catch is significant in **every** family
(CI-low 0.16–0.62, pooled **71/120** residual); only the 3B `llama32_3b` is the dead verifier never
added, on all 6 families that run the greedy. Two nuances the new families surface, both *supporting*
the thesis: (i) MBPP nudges the lower bound to **1.06** (`ρ_v=0.938`, the most redundant family — code
self-verification saturates hardest); (ii) MBPP is the first **role-flip** — `mistral7b` is the
strongest *solo* verifier (code-specialised) while `llama31`/`qwen7b` diversify, i.e. *which* model is
best is domain-dependent but a cross-family verifier still adds real catch. No 4-model joint pool is
run: the greedy already draws candidates from the full 4-model universe and stops (Δcatch CI-low→0)
before a 4th is worth adding, so the joint pool adds no inferential content beyond the marginals.

### 3.5 Controller invariant — the framing payoff (C6) **[done — NEW]**
The greedy controller (`theory/marginal_value.py`) adds a candidate **only when its Wilson CI-low
on Δcatch > 0**. Across all 4 MCQA families the invariant holds: the dead 3B verifier
`llama32_3b` (zero unique catch) is **correctly never diversified in**, while `llama31` — dead at
the small-n pilot — now shows a *significant* marginal catch at n=150 and is added on 3/4 families
(it is even the solo START on TruthfulQA). This is itself the point: *no fixed model-identity or
family label decides who to add — the controller must measure conditional marginal catch on the
current task*, and the same label flips between "add" and "don't add" as the data grow.
Benign artifact documented: phase-diagram `S` cells (TruthfulQA) appear only from a degenerate
single-dead-verifier seed, not a genuine scale recommendation.

### 3.6 Context confound locked (C7) **[done]**
Evidence-preserving context mode (`--context-mode evidence`, no summariser), seeds held fixed so
context is the *only* variable:

| mode | n_eff (trunc→evid) | blind (trunc→evid) | β_context |
|---|---|---|---|
| FM-3.1 | 1.82 → 1.75 | 0.63 → 0.72 | −0.09 |
| FM-3.2 | 1.89 → 1.89 | 0.95 → 0.95 | +0.00 |

`n_eff` is context-invariant and `β_context ≈ 0` ⇒ the *correlation* structure is not a truncation
artifact. (This does **not** by itself establish an intrinsic blind-spot atom: the ceiling `π₀` is
separately **not statistically identified** — see §3.9, C11 — so we do not claim blind spots are
intrinsic, only that context does not move `n_eff`.)

### 3.7 Cascade Auditor (C8) **[done]**  `theory/auditor.py`
Ingest a pool (or a small probe) → per-verifier self-`n_eff` (SCALE redundancy), pool ordering,
`k*`, reliability, ceiling, **Verification Debt = calls − n_eff**, **Efficiency = n_eff/calls**, and
a Scale/Diversify/Stop/Escalate recommendation with a "adding N more calls is statistically useless"
warning. On the real pool: recommends ESCALATE when the target exceeds the 0.86 ceiling, STOP when
met at `k*=3`.

### 3.8 Methodology freeze (C9) **[done — NEW]**  `docs/METHODOLOGY_FREEZE.md`
Locked knobs: canonical population = errors-only `n_gates==50`; `ctx_truncate`; classifier threshold
`0.5`; greedy `λ=0.1`; add-rule Wilson CI-low(Δcatch)>0; redundancy = Beta-Binomial `n_eff@50`;
universe `{llama31, llama32_3b, mistral7b, qwen7b}`. Interpreter `~/venvs/env/bin/python`
(`.venv` is broken). **No further controller changes or tuning.**

### 3.9 Blind-spot `π₀` re-fit on large-n (C11) **[secondary — CLOSED, bounded negative result]**
`scripts/blind_spot_largen.py` re-fits the ceiling mixture (M3 atom vs M2 no-atom) on the n=150 data;
`scripts/blind_spot_power.py` is the pre-registered power analysis. Result: on **0/5** families does
same-model `π₀` clear the bar (profile `CI₉₅` excludes 0 **and** boundary-LRT `p<0.05`). The raw
ceiling mass is real (ARC 0.08, MMLU 0.11, CSQA 0.08, TruthfulQA 0.36, MAST 0.70) but a heavy
Beta-Binomial tail (ρ_v≈0.71–0.85) already explains it — the atom is **not identified**.

| same-model | π̂₀ | CI₉₅(π₀) | atom-LRT p | verdict |
|---|---|---|---|---|
| ARC | 0.073 | [0, 0.125] | 0.116 | upper bound |
| MMLU | 0.091 | [0, 0.155] | 0.148 | upper bound |
| CSQA | 0.073 | [0, 0.125] | 0.106 | upper bound |
| TruthfulQA | 0.258 | [0, 0.400] | 0.276 | upper bound |
| MAST-3.3 | 0.000 | [0, 0.760] | 1.000 | upper bound |

> **Locked conclusion (paper wording).** *Blind-spot mass was not statistically identified. We
> therefore report upper bounds rather than claim the existence of a non-zero shared-failure atom.*
> **Methodological observation.** *A high apparent blind-spot mass does not imply that the atom is
> identifiable, because persistent failures can be statistically confounded with a heavy
> correlated-error tail.*

**Power (if the fitted atom is *real*):** ~500 items ⇒ 0.8 power for ARC/CSQA, ~900 for MMLU, but
**TruthfulQA stays <0.8 even at 2000** despite the largest π̂₀ — its residual tail is the heaviest,
so the atom is maximally confounded. This is an **identifiability wall, not a sample-size wall**:
300 items is useless everywhere, 500 is a coin-flip that only pays if the atom is genuinely there.
**Decisions:** (a) do *not* brute-force item count under truncate; (b) the informative next lever is
**evidence-preserving context** (π₀ is context-sensitive; truncation may be manufacturing the accept
tail); (c) for the paper, report the π₀ **upper bounds** and lead the secondary story with the clean,
atom-free **diversification contrast** — raw shared-failure mass collapses `same→cross-family`
(MAST 0.70→0.00, MMLU 0.11→0.01, TruthfulQA 0.36→0.02), which needs no atom identification.

### 3.10 Fusion-rule ablation — the controller tracks the cost-optimal rule (C12) **[done — NEW 2026-08-11]**
`scripts/ablation_fusion_baselines.py` (offline, no GPU) benchmarks the frozen controller against the
standard fusion rules a reviewer would reach for, on the **same** per-item verdicts (the frozen
majority-reject definition, `theory.marginal_value._prep`), over the **6 identical-setup confirmatory
families** (MCQA×4 + GSM8K + MBPP; MAST-3.3 excluded — non-rectangular verifier×item matrix). Metric:
utility `U = catch − ρ·fa` (catch = TPR on errors, fa = FPR on clean, `ρ` = false-alarm/miss cost).
Tables: `results/tables/fusion_{baselines,scale_panel,cost_sweep}.csv`.

**(a) Repetition is worthless, diversity is not — in fusion terms.** Adding an independent *same-model*
repeat (half-split 2nd draw) lifts error-catch by **+0.003** (macro-avg over 6 families); adding a
*diverse* verifier lifts it by **+0.079** (~26×). This is `n_eff@50 ≈ 1` restated as a fusion result.

**(b) No universal fusion rule — the winner is cost-driven, and the controller follows it.** Run the
greedy controller *at* each cost ratio (`greedy_marginal_value(df, w_fa=ρ)`, OR-cascade fusion of the
selected set) and compare to the best *fixed* rule:

| `ρ` = fa/miss | best fixed rule | best-fixed `U` | **controller `U`** | controller set-size | tracks? |
|---|---|---|---|---|---|
| 0.0 (miss-dominated) | OR / cascade | 0.954 | 0.953 | 2.67 | ✅ |
| 0.25 | pattern-Bayes stack | 0.761 | 0.761 | 2.33 | ✅ |
| 0.5 | pattern-Bayes stack | 0.644 | 0.588 | 2.17 | ✱ |
| 1.0 (balanced) | single-best verifier | 0.475 | 0.460 | 1.50 | ✅ |
| 2.0 | MAJORITY | 0.275 | 0.259 | 1.00 | ✅ |
| 4.0 (fa-dominated) | AND / unanimous | 0.093 | −0.034 | 1.00 | ✦ |

The optimal fixed rule **flips** `OR → stack → single-best → MAJORITY → AND` as false alarms get
costly — there is **no universal fusion rule**. The controller **tracks the cost-optimal rule at 4/6
cost ratios with one knob** (no per-cost re-tuning), and its selected set **shrinks 2.67 → 1.0
verifiers** as `ρ` grows. Two honest gaps: (✱) at `ρ=0.5` a *supervised* pattern-Bayes stack squeezes
+0.056 the fixed cascade can't — label-trained headroom, a future-work fusion head; (✦) at `ρ=4` the
optimum flips to **conjunctive AND-fusion**.

**Scope boundary (deliberate, = ESCALATE).** The cascade is **disjunctive** ("reject if *any* gate
rejects"), which only *adds* false alarms; in the **precision-critical regime** (`ρ ≳ 4`) the
cost-optimal fusion is **conjunctive** (require unanimity), a topology the reject-any cascade cannot
represent at any set size. This is exactly the boundary the 4th controller decision names: past the
disjunctive ceiling you **ESCALATE** — switch fusion mode / go out-of-pool — rather than keep adding
gates. So the one regime the controller does not track is not a failure but the documented hand-off
point of the framework.

> **Honest framing.** The claim is **not** "OR-cascade is the best fusion rule" — at the balanced
> point (`ρ=1`) naive OR of all 4 verifiers is *dominated* by a selective rule. The claim is that the
> controller **selects the cost-appropriate operating point**, which is the whole reason it exists.

---

## 4. Key results at a glance

```
  same-model self-verification saturates       →  n_eff@50 = 1.09 – 1.89   (never 50)
  cross-family / diversity                      →  n_eff@50 = 2.35 – 5.24, can drive blind-spot 0.95→0
  redundancy null replicated across 7 families  →  n_eff@50 ∈ [1.06, 1.21]   (large-n, n=150; +GSM8K reasoning, +MBPP code)
  controller invariant (add ⇔ Wilson CI-low>0)  →  holds 6/6 greedy families; dead 3B llama32_3b never added
  diversify signal (now POWERED)                →  per-family CI-low>0 (0.16–0.62), pooled 71/120 residual
  blind-spot π₀ (secondary)                     →  NOT established: 0/5 families, CI touches 0; report upper bounds ≤0.13–0.40
  ↳ why (identifiability, not n)                 →  ρ_v≈0.85 tail explains ceiling; TruthfulQA π̂₀=0.26 still <0.8 power @2000
  diversification breaks shared failures         →  raw ceiling collapses same→cross-family (MAST 0.70→0.00) — atom-free, clean
  fusion-rule ablation (no universal rule)       →  winner flips OR→stack→single→MAJORITY→AND with fa/miss cost; controller tracks it 4/6, one knob
  SCALE vs DIVERSIFY in fusion terms             →  same-model repeat Δcatch +0.003 vs diverse verifier +0.079 (~26×)
  disjunctive-cascade boundary                   →  ρ≳4 (precision-critical) needs conjunctive AND-fusion = ESCALATE, out of the reject-any family
```

---

## 5. Roadmap  **[NEXT = paper write-up on the fusion axis]**

- **C10 Confirmatory large-n [DONE 2026-08-10].** Re-ran the *frozen* pipeline at n=150 err/family
  (150×50 gates × 3 pools, 11250 backend calls/family, 0 unparsed, 0 overflows). Redundancy null held
  (n_eff@50 ∈ [1.09,1.21]); the diversify signal is now **powered** (residual 1–6 → 15–36 per family,
  every family CI-low>0). No re-tuning.
- **C10b Reasoning + code extension [DONE 2026-08-11].** Ran the same *frozen* pipeline on **GSM8K
  (reasoning)** and **MBPP (code)** — 11,250 calls each, 0 unparsed / 0 overflow. Both **HELD**:
  same-model `n_eff@50` 1.16 / 1.06 (MBPP the most redundant family yet, `ρ_v=0.938`, widening the
  null band to **[1.06,1.21]** across seven families), diversify CI-low>0 (mistral7b 0.16 on GSM8K;
  llama31 0.22 + qwen7b 0.05 on MBPP), invariant holds (`llama32_3b` never added). MBPP is the first
  **role-flip** — mistral7b is the strongest *solo* on code — which is the thesis, not a violation:
  the best verifier is domain-dependent, cross-family still adds catch. `redundancy_invariant_7family.csv`.
- **C11 Blind-spot `π₀` (secondary) [DONE — bounded, not established].** Re-fit on the large-n data
  already collected (`scripts/blind_spot_largen.py`) + pre-registered power (`blind_spot_power.py`).
  Outcome: **0/5** families clear existence — CI touches 0 and the atom-LRT is n.s. everywhere; a
  ρ_v≈0.71–0.85 Beta-Binomial tail already accounts for the ceiling mass. Power analysis shows this
  is an **identifiability wall, not sample size**: 300 items is useless, 500 only reaches 0.8 for
  ARC/CSQA *and only if the atom is genuinely real*, and TruthfulQA never clears 0.8 through n=2000.
  **Next (no blind GPU):** re-measure π₀ under **evidence-preserving context** (`--context-mode
  evidence`) — π₀ is context-sensitive, and if truncation manufactures the accept tail, evidence mode
  either kills the claim cleanly (blind spot was observational) or thins the tail enough to identify a
  real atom at the *current* n. Paper stance: report π₀ **upper bounds** and carry the secondary story
  with the atom-free **diversification contrast** (raw shared-failure mass `same→cross-family`
  collapses in all five families) — *repeated verification cannot eliminate systematic shared
  failures; diversification can break what scaling the same verifier cannot* stands descriptively.
- **Optional** — scale the 5th family (MAST-3.3) via `--repeats`, same archive caveat.
- **Paper** — freeze figures (fig7 SDSE map flagship, fig8 effective verifiers), reconcile the
  base-theory citation (Han, arXiv:2607.13918), submit to Information Fusion.

---

## 6. Repository layout

```
theory/        estimator + proofs-as-code
  betabinom_mixture.py   Beta-Binomial/ceiling MLE → ρ_v, n_eff, π₀
  marginal_value.py      greedy marginal-catch controller + phase diagram
  net_utility.py         U_λ = R − λC frontier, k*
  auditor.py             the Cascade Auditor (C8)
  probe.py               2-verdict probe (Thm 3)
  cascade_design.py · odds_law.py · ceiling_test.py · concavity_test.py · trichotomy_test.py
cascade/       data collection
  gate_runner.py         k serial gates per item → JSONL  (NB: append mode)
  mast_loader.py · mcqa_loader.py    the two item sources (identical polarity)
  verifier_chain.py · ollama_backend.py · alpha_recorder.py · paths.py
scripts/       analysis & launchers (NOT the controller)
  pilot_mcqa.py · pilot_gsm8k.py · pilot_mbpp.py   per-bench greedy + phase + n_eff
  freeze_redundancy_invariant.py            merged 5-family table (C5/C6, frozen artifact)
  freeze_redundancy_invariant_7family.py    +GSM8K reasoning +MBPP code → 7-family table
  ablation_fusion_baselines.py     C12 fusion-rule ablation + cost sweep + controller overlay
  run_mcqa_confirmatory.sh         C10 launcher (archive-first)
  run_mcqa_benchmarks.sh · freeze_main_table.py · context_decomposition.py · ...
docs/          paper_draft.md · theory_cascade_design.md · METHODOLOGY_FREEZE.md · related_work.md
results/tables/  main_results_3mode · redundancy_invariant_{5,7}family · {gsm8k,mbpp}_marginal · fusion_{baselines,scale_panel,cost_sweep} · mcqa_{marginal,phase}_* · ...
tests/         62 passing
```

---

## 7. Install & run

```bash
PY=~/venvs/env/bin/python                # .venv is broken; use this interpreter
$PY -m pytest -q                         # 62 tests

# --- data collection (needs local Ollama serving the small verifiers) ---
$PY run_cascade.py --source mcqa --benchmark arc --pool same_model \
    --k 50 --n-errors 40 --n-non-errors 20 --skip-analyse

# --- offline analysis (no GPU) ---
$PY scripts/pilot_mcqa.py                     # per-bench greedy / phase / n_eff
$PY scripts/pilot_gsm8k.py                    # GSM8K reasoning family
$PY scripts/pilot_mbpp.py                     # MBPP code family
$PY scripts/freeze_redundancy_invariant.py           # frozen 5-family redundancy + invariant table
$PY scripts/freeze_redundancy_invariant_7family.py   # +GSM8K +MBPP → 7-family table
$PY scripts/ablation_fusion_baselines.py             # C12 fusion-rule ablation + cost sweep + controller overlay

# --- confirmatory large-n (C10) — long run, launch detached ---
nohup scripts/run_mcqa_confirmatory.sh > confirmatory_mcqa.log 2>&1 &
echo $! > confirmatory_mcqa.pid;  tail -f confirmatory_mcqa.log
```

---

## 8. Core objects

```
per-item stat     (m_i, K_i) = unwanted-verdict count over decided gates
correlation       ρ_v = 1/(a+b+1)            (Beta-Binomial)
effective count   n_eff(K) = K / (1 + (K−1)ρ_v)
ceiling           R(S) ≤ 1 − π₀              (Thm 2)
objective         U_λ = R − λC               greedy (1−1/e)-optimal (Thm 1)
add-rule          add candidate ⇔ Wilson CI-low(Δcatch) > 0     (frozen)
stop-rule         stop at first u_t = Δ_t − λc_t ≤ 0             (Thm 4)
debt / efficiency Verification-Debt = calls − n_eff ; Efficiency = n_eff / calls
```

## 9. Limitations
1. **Verifier scope (stated, not apologised for).** The pool is **5 checkpoints across 3 pretraining
   families (Qwen, Llama, Mistral) and 2 size tiers (3B–14B)** — enough distinct families to show the
   redundancy/diversity effect is not a single-architecture artifact, but all **open and ≤14B**. We do
   **not** claim frontier/closed models (GPT/Claude) or other families (Gemma/Phi/…) behave
   identically. These small verifiers also have high miss rates on FC3 (weak verifiers) — itself a
   finding.
2. Diversify/selection gain is small on real pools (Thm 6 explains it); the value is the framework +
   guarantees + STOP/DIVERSIFY/ESCALATE + the auditor, **not** an accuracy jump.
3. Positive diversify signal is now powered at n=150 (per-family CI-low>0); redundancy null remains
   the strongest result. The **blind-spot `π₀`** is a secondary *negative* result (closed): the atom
   was **not statistically identified** (CI touches 0, atom-LRT n.s. on all 5 families), so we report
   an upper bound, not existence — and note it is confounded with the heavy correlated-error tail.
4. `n_eff` assumes conditional independence given the item (supported: verdict autocorrelation ≈ 0,
   product formula predicts held-out outcomes to ≈ 0.03) but not proven for all pools.
5. **Disjunctive scope (deliberate boundary, = ESCALATE).** The cascade is *reject-any* (disjunctive),
   so it only *adds* false alarms; in the precision-critical regime (`ρ≳4`, fa ≫ miss) the cost-optimal
   fusion is *conjunctive* (unanimity), which the cascade cannot represent (§3.10). This is not a bug
   but the framework's documented hand-off: past the disjunctive ceiling you **ESCALATE** (switch
   fusion mode / go out-of-pool), not keep adding gates. A learned fusion head is future work.

---

## 10. Full work checklist  *(every task from infra → pipeline close; [done] shipped · [todo] pending)*

| # | Phase | Task | Status |
|---|---|---|---|
| 1 | Infra | Testbed integration — `shared_testbed` link, MAST loader, MCQA loader (identical polarity) | **[done]** |
| 2 | Infra | Verifier chain + gate runner (k serial gates → JSONL), Ollama backend, 5 verifier models | **[done]** |
| 3 | Infra | Canonical population = errors-only `n_gates==50`; k20-pilot contamination bug fixed (π₀ 0.356→0) | **[done]** |
| 4 | Estimator | Beta-Binomial / ceiling-mixture MLE → `ρ_v`, `n_eff(K)`, blind-spot `π₀` | **[done]** |
| 5 | Theory | 6 theorems + proofs (`docs/theory_cascade_design.md`): submodular greedy, ceiling, probe, stopping ×2, value≠accuracy | **[done]** |
| 6 | Theory | Controller modules — `marginal_value` (greedy + phase), `net_utility`, `auditor`, `probe`, `cascade_design` | **[done]** |
| 7 | Theory | 62 unit tests passing | **[done]** |
| 8 | Empirics | MAST FC3 3-mode main results, frozen (`main_results_3mode.csv`) | **[done]** |
| 9 | Empirics | Context confound locked — evidence mode, `β_context ≈ 0` (correlation not a truncation artifact; π₀ atom not claimed) | **[done]** |
| 10 | Empirics | `same_family` dead-verifier diagnosis (n_eff=7.39 artifact → use marginal-catch) | **[done]** |
| 11 | Empirics | MCQA pilot data collection — ARC/MMLU/CSQA/TruthfulQA × 3 pools (raw runs) | **[done]** |
| 12 | Empirics | MCQA analysis tables — `pilot_mcqa.py` → `mcqa_{marginal,phase}_*.csv` | **[done]** |
| 13 | Empirics | 7-family redundancy replication table (`redundancy_invariant_{5,7}family.csv`, +GSM8K +MBPP, n_eff∈[1.06,1.21]) | **[done]** |
| 14 | Empirics | Controller invariant verified 6/6 greedy families (dead 3B `llama32_3b` never added) | **[done]** |
| 15 | Freeze | Methodology + hyperparameters locked (`docs/METHODOLOGY_FREEZE.md`, 2026-08-09) | **[done]** |
| 16 | Figures | `fig1–fig8` generated (fig7 SDSE flagship, fig8 effective verifiers) — pilot numbers | **[done]** |
| 17 | Docs | README refreshed to frozen state; claim-discipline guardrails embedded | **[done]** |
| 18 | Confirmatory | Large-n MCQA run — n=150 err/family, 3 pools, 11250 calls/family, 0 unparsed | **[done — 2026-08-10]** |
| 19 | Confirmatory | Re-analyse large-n → refreeze `mcqa_*` + `redundancy_invariant_5family` tables (n_eff∈[1.09,1.21], diversify 60/93) | **[done]** |
| 20 | Blind-spot | Re-fit ceiling `π₀` on large-n → **0/5 identified**; report upper bounds (`scripts/blind_spot_largen.py`) | **[done]** |
| 21 | Blind-spot | Power analysis → identifiability wall (TruthfulQA <0.8 @ n=2000); **decision: no more GPU** (`blind_spot_power.py`) | **[done]** |
| 22 | Blind-spot | **CLOSED as secondary negative result** — locked wording in §3.9; evidence-mode re-measure optional, not blocking | **[closed]** |
| 23 | Confirmatory | (Optional) scale 5th family MAST-3.3 via `--repeats`, same archive caveat | **[todo]** |
| 28 | Confirmatory | Extend to reasoning+code — GSM8K & MBPP large-n (n=150, 11250 calls each, 0 unparsed/overflow) → 7-family table; both HELD (`freeze_redundancy_invariant_7family.py`) | **[done — 2026-08-11]** |
| 29 | Ablation | Fusion-rule ablation (C12) — controller vs OR/AND/majority/Dawid-Skene/weighted/pattern-Bayes; no universal rule, controller tracks cost-optimal 4/6, conjunctive-AND→ESCALATE boundary (`ablation_fusion_baselines.py`) | **[done — 2026-08-11]** |
| 24 | Confirmatory | Regenerate `fig1–fig8` on large-n numbers | **[todo]** |
| 25 | Paper | Reconcile base-theory citation (Han, arXiv:2607.13918 — earlier mislabelled "Aksu 2026b") | **[todo]** |
| 26 | Paper | Finalize `docs/paper_draft.md` (abstract → limitations) with confirmatory numbers | **[todo]** |
| 27 | **Close** | Submit to Information Fusion (Q1) → **pipeline closed** | **[todo]** |
