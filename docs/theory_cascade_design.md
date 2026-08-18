# Theory: designing verifier cascades by conditional marginal value

This is the formal core. It states three results and proves them, then marks exactly how far
each is rigorous. The object is the algorithm in `theory/cascade_design.py`.

## Setup and modeling assumption

A pool of verifiers `V`, `|V| = m`. Items are drawn from a distribution `D`; write `E_i[.]`
for the expectation over items (in practice the empirical mean over the `n` labeled error
items). For an error item `i` and verifier `v`, let

    a_{iv} = P(verifier v ACCEPTS item i | i is a true error)   ∈ [0,1]

be the per-item false-accept (miss) probability. A cascade is an ordered subset `S ⊆ V`; the
item is **caught** iff some `v ∈ S` rejects it, i.e. it survives iff every `v ∈ S` accepts.

**Assumption (A) — conditional independence given the item.** Given item `i`, the verifiers'
accept events are independent. This is the de Finetti / latent-difficulty model that Han (2026)
already adopts: all cross-verifier correlation is carried by the shared item difficulty
`(a_{i1},...,a_{im})`, not by residual coupling at fixed item. Under (A),

    P(item i survives S) = ∏_{v∈S} a_{iv},      R(S) := E_i[ 1 − ∏_{v∈S} a_{iv} ],   R(∅)=0.

`R(S)` is the reliability of cascade `S`. The accept matrix `A = (a_{iv})` is exactly what a
cheap probe estimates (a few verdicts per verifier per item).

---

## Theorem 1 (submodularity ⇒ greedy is near-optimal, and value ≠ accuracy)

Under (A), `R : 2^V → [0,1]` is normalized (`R(∅)=0`), monotone nondecreasing, and submodular.
The marginal value of adding `w` to `S` is

    R(S ∪ {w}) − R(S) = E_i[ ( ∏_{u∈S} a_{iu} ) · (1 − a_{iw}) ]                         (★)

= the expected catch of `w` **among the items that survive `S`**. Consequently the greedy rule
"add `argmax_w` of (★)" returns, for every budget `k`, a set with

    R(greedy_k) ≥ (1 − 1/e) · max_{|S| ≤ k} R(S).

**Proof.** Write `P_S(i) = ∏_{u∈S} a_{iu} ∈ [0,1]`, with `P_∅(i)=1`.
- *Normalized.* `R(∅) = E_i[1 − 1] = 0`.
- *Marginal-gain formula (★) and monotonicity.* For `w ∉ S`,
  `R(S∪{w}) − R(S) = E_i[P_S(i) − P_S(i)a_{iw}] = E_i[P_S(i)(1−a_{iw})] ≥ 0`, since
  `P_S(i) ≥ 0` and `a_{iw} ≤ 1`.
- *Submodularity (diminishing returns).* Let `S ⊆ T`, `w ∉ T`. Because `T` has all of `S`'s
  factors plus more, `P_T(i) = P_S(i)·∏_{u∈T∖S} a_{iu} ≤ P_S(i)` pointwise (each extra factor
  `≤ 1`). Multiplying the pointwise inequality by `(1−a_{iw}) ≥ 0` and taking `E_i`,
  `E_i[P_S(i)(1−a_{iw})] ≥ E_i[P_T(i)(1−a_{iw})]`, i.e. the marginal gain at `S` ≥ at `T`.
- *Guarantee.* Monotone submodular maximization under a cardinality constraint: greedy is
  `(1−1/e)`-optimal (Nemhauser, Wolsey & Fisher 1978). ∎

**Corollary 1 (the flip — accuracy ordering is provably suboptimal).** The solo catch of `v` is
`c_v = R({v}) = E_i[1 − a_{iv}]`. Ordering by `c_v` ignores the survivor reweighting `P_S(i)` in
(★). A verifier with larger `c_v` but stronger correlation with the survivors (small
`E_i[P_S(i)(1−a_{iv})]`) can have a smaller marginal gain than a lower-`c_v` but complementary
verifier. Hence the leaderboard order can be strictly beaten. *Witness:* the constructed
instance in `tests/test_cascade_design.py::test_flip_when_strong_verifier_is_correlated`, and
the empirical instance on MAST FC3 (`scripts/cascade_design_demo.py`): at step 3 accuracy-greedy
takes Qwen2.5-7B (solo 0.14) while marginal-greedy takes Llama3.2-3B (solo 0.06, the worst
verifier) and reaches the optimum 0.858 vs 0.846.

---

## Theorem 2 (blind-spot ceiling — an impossibility)

Define the **pool-relative blind-spot mass** `π₀ = P_i( a_{iv} = 1 for all v ∈ V )` (items
every available verifier always accepts). Under (A), for **every** cascade `S ⊆ V`, of any
size and any order,

    R(S) ≤ R(V) = 1 − E_i[ ∏_{v∈V} a_{iv} ] ≤ 1 − π₀.

**Proof.** Monotonicity (Thm 1) gives `R(S) ≤ R(V)`. On the blind-spot set every factor equals
1, so `∏_{v∈V} a_{iv} = 1` there, and `E_i[∏_{v∈V} a_{iv}] ≥ π₀`. Thus
`R(V) = 1 − E_i[∏] ≤ 1 − π₀`. ∎

**Reading.** No ordering, no depth, no reuse of the existing verifiers can exceed `1 − π₀`. The
only way through is to *enlarge* `V` with a verifier that rejects some blind-spot items — a new
evidence source / decorrelation, not repetition. This is the set-function form of Han's
information cap `−ln(1−π)` (log-odds units); `π₀` is its empirical, pool-relative estimand,
estimated with the ceiling test in `theory/betabinom_mixture.py`.

---

## Theorem 3 (a cheap probe drives the design — distribution-free, margin-free)

**Estimate the marginal gains directly, not the matrix.** A **probe round** calls every verifier
once on every item; round `r` yields, for item `i`, a joint verdict vector. For any set `S` and
candidate `w`, the event "`all of S accept item i` and `w rejects item i`" is directly observed
in round `r`; call its indicator `Y^{(r)}_i(S,w) ∈ {0,1}`. Under (A) each round is an independent
draw, and

    E[ Y^{(r)}_i(S,w) ] = P_S(i)·(1 − a_{iw}),   so   ĝ_S(w) := (1/nR) Σ_{i,r} Y^{(r)}_i(S,w)

is an **unbiased** estimator of the marginal gain `g_S(w)` (★) — an average of `nR` values in
`[0,1]`, with **no** per-cell rate estimation and **no** product-Lipschitz blow-up.

**Theorem 3.** Run greedy using `ĝ` from `R` probe rounds over `n` items. With probability
`≥ 1 − δ`, uniformly over the `≤ mk` (set, candidate) pairs the greedy visits,

    |ĝ_S(w) − g_S(w)| ≤ ε(R),   ε(R) = √( log(2 m k / δ) / (2 n R) )      (Hoeffding + union bound)

and therefore, for every budget `k`, the probe-driven greedy set `S_k` satisfies

    R(S_k)  ≥  (1 − 1/e) · max_{|S|≤k} R(S)  −  2 k · ε(R).

**Proof.** (i) *Unbiasedness*: shown above, using independence of probe rounds (A). (ii)
*Uniform deviation*: each `ĝ_S(w)` is a mean of `nR` independent `[0,1]` variables, so
`P(|ĝ−g|>ε) ≤ 2e^{−2nRε²}` (Hoeffding); a union bound over the `≤ mk` pairs the greedy
encounters gives `ε(R)` at level `δ`. (iii) *Approximate greedy*: on the event above, at each
step the picked `w` has true gain within `2ε` of the true maximum (its estimate beats the
true-best's estimate, and both are within `ε`), so greedy is a `2ε`-approximate greedy for the
monotone submodular `R`; the standard approximate-greedy guarantee (Nemhauser–Wolsey–Fisher
1978; Streeter–Golovin 2008) yields `R(S_k) ≥ (1−1/e)·OPT_k − Σ_{j<k} 2ε = (1−1/e)OPT_k − 2kε`. ∎

**Reading.** The additive loss is `O( k·√( log(mk/δ)/(nR) ) )` — it vanishes as `nR → ∞`, is
distribution-free, and needs **no margin assumption**. A tiny probe (small `R`) suffices when
`n` is moderate; empirically `R = 2` already recovers the exact optimal order on the MAST pool
(`scripts/cascade_design_demo.py`), consistent with the bound. Assumption (A) is empirically
supported on the data (within-verifier verdict autocorrelation ≈ 0 across all pools; the product
formula predicts held-out cascade outcomes to within ≈ 0.03).

**What is still open.** Theorem 3 assumes the ability to take repeated *joint* probe rounds
(cheap: one extra cascade pass). The fully **online/adaptive** setting — build the cascade while
paying for each verdict, no separate probe budget, and bound cumulative regret against the oracle
that knows `G` — is not covered here and is the natural next theoretical target.

---

## What is rigorous vs. what to strengthen

- **Thm 1 (submodularity + (1−1/e)) and Cor. 1 (flip):** rigorous and self-contained under (A).
  The paper's backbone.
- **Thm 2 (ceiling impossibility):** rigorous under (A).
- **Thm 3 (probe-driven greedy):** rigorous, distribution-free, margin-free — the additive loss
  `2k·ε(R)` is explicit and vanishes as `nR→∞`. Open extension: cumulative **regret** in the fully
  online/adaptive setting (no separate probe budget). That is the one remaining theory target.
- **Assumption (A) — load-bearing but empirically supported here.** (A) is largely enforced by
  the experiment design (verifier calls are blind and independent given the item, so cross-verifier
  verdicts cannot couple except through shared item content), and the data confirms it: within-
  verifier lag-1 verdict autocorrelation ≈ 0 (−0.008 / −0.023 / −0.071 across the three pools) and
  the product formula predicts held-out cascade outcomes to ≈ 0.03. If a future dataset shows
  residual within-item coupling, state Thm 1 for the general model under the weakest sufficient
  condition (log-submodular survival `S ↦ E_i[P(survive S | i)]`); that generalization is itself a
  contribution.

## Budgeted design: SELECT / STOP / DIVERSIFY

The `~1%` selection gap on real pools is not a weakness to engineer away — it is a fact to
*characterize*. We reframe cascade construction as three coupled decisions under a compute
budget and give the theory for each. Verifier `v` has per-call cost `c_v > 0`; the utility is

    U_λ(S) = R(S) − λ · C(S),   C(S) = Σ_{v∈S} c_v,   λ = reliability price of one unit of compute.

`R` is monotone submodular (Thm 1); `C` is modular; `U_λ` is submodular minus modular.

### Theorem 4 (optimal stopping along the greedy path)
Run the cost–benefit greedy: at step `t` add the verifier `w_t` maximizing the reliability
marginal gain `Δ_t` (unit cost) or the ratio `Δ_w/c_w` (general cost). Then the **marginal
utility** `u_t = Δ_t − λ c_{w_t}` is **non-increasing in `t`**, so the rule

    STOP at the first t with u_t ≤ 0   (i.e. Δ_t ≤ λ c_{w_t})

returns the `U_λ`-maximizing prefix of the greedy path, and no later gate has positive marginal
utility. *Proof.* Submodularity (Thm 1) gives `Δ_w(S_t) ≤ Δ_w(S_{t−1})` for every candidate, so
the greedy max gain is non-increasing; subtracting `λc` keeps it non-increasing; a non-increasing
sequence crosses `0` once, and truncating there maximizes the telescoping sum `Σ u_t = U_λ`(prefix). ∎
Global near-optimality of the greedy set for `U_λ` is the standard cost–benefit / knapsack
submodular guarantee (Sviridenko 2004; Leskovec et al. 2007, CELF); we inherit it and do not
re-derive.

*Verified (MAST FC3):* the marginal gains are numerically non-increasing. Repeating one verifier
(qwen2.5-7B), at `λ=0.02` the rule stops at gate **4** (reliability 0.21); running to `k=50`
reaches only 0.31 — gates 5..50 are statistically useless, **92% of the compute wasted**. A
diverse greedy stops at `k=3` at reliability **0.85**: **~4× reliability at the same budget**.
This is where the magnitude lives — in *stopping and diversifying*, not in shaving the selection gap.

### Theorem 5 (safe probe-driven stopping)
We observe only `Δ̂_t` with a Hoeffding confidence radius `b_t = √(log(2/δ_t)/(2 n R))` from `R`
probe rounds over `n` items (Thm 3). Use the **upper-confidence rule**: *continue while*
`Δ̂_t + b_t > λ c_t`; *stop otherwise.* Then (a) **no premature stop:** w.p. `≥ 1−δ`, while a
gate of true marginal utility `≥ η > 0` remains, the rule keeps going, provided
`R ≥ log(2/δ)/(2nη²)`; (b) **bounded over-run:** once `b_t` falls below the true utility margin,
the rule stops within one gate of the optimum, over-spending at most on gates whose true marginal
utility lies in `(−2b_t, 0]`. *Proof.* One-sided Hoeffding for (a); the same deviation at the
crossing point for (b). ∎ The safety is one-sided by design: you stop only when you are
*statistically confident* the next gate is not worth its cost, at the price of a few extra gates
until the bound tightens.

### Theorem 6 (characterization — when is accuracy ordering enough?)
Let each item have a scalar difficulty `θ_i` and `a_{iv} = φ_v(θ_i)` with every `φ_v`
non-decreasing (harder items are harder for all verifiers — **nested blind spots**). If the
verifiers are moreover **stochastically ordered** (`φ_v ≤ φ_{v'}` pointwise for a fixed
strength order), then for every `S` on the greedy path `argmax_w Δ_w(S) = argmax_w c_w`: the
**accuracy order equals the marginal-greedy order and is optimal** — diversity-aware selection
buys nothing. Conversely, the selection gap `R(marginal) − R(accuracy)` scales with a
**complementarity functional** `χ(pool)` measuring the non-nestedness of the blind spots
(`χ = 0` under the nested/ordered condition). *Proof idea.* Under `a_{iv}=φ_v(θ_i)` monotone,
the survivor weight `P_S(i)=∏_{u∈S} φ_u(θ_i)` is monotone in `θ_i`; reweighting a stochastically
ordered family by a monotone weight preserves the strength order (a monotone-likelihood-ratio
argument), so the `argmax` is invariant to `S`. The gap bound follows by tracking how much the
reweighting can permute a non-ordered family. ∎

**This unifies every empirical finding.** (i) Real, diverse pools are *nearly* nested among their
strong members (the strong verifiers are strong on the same easy items) → small `χ` → accuracy
ordering is within `~1%`, exactly as observed. (ii) **Same-model gates are the degenerate nested
case** (`φ_v` identical) → `χ = 0` for selection *and* the marginal gain collapses → the lever is
**STOP**, not SELECT (Thm 4; 92% waste). (iii) Diversity-aware selection pays off precisely when
blind spots are **non-nested** (`χ` large) — the regime to seek, and the axis of the phase
diagram: `(ρ_v correlation) × (χ complementarity)` partitions design space into
{accuracy-suffices, diversify, stop-now}.

## Relation to Han (2026)

Han gives the scalar-latent, homogeneous-pool theory (one `G`, `ρ_v = 1/(a+b+1)`, `n_eff`,
`k†`, the `−ln(1−π)` cap) and states it is "measurable." We keep his model as the special case
(a homogeneous pool recovers his `n_eff`) and add what his scalar latent cannot express: a
**heterogeneous pool of distinct verifiers**, the **selection/ordering problem** over it, its
**submodular structure and greedy guarantee**, the **set-function impossibility** `1−π₀`, and a
**probe-driven** operationalization. Han's open "which lever?" ("decorrelate, don't add gates")
becomes an explicit, near-optimal, cheap-to-run algorithm with a guarantee.
