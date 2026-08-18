# Framing for Information Fusion (Q1)

**Title:** *Budget-Aware Fusion of Correlated LLM Verifiers: When to Scale, Diversify, Stop, or
Escalate.*

**Reframe.** Treat each LLM verifier not as an "LLM-as-a-judge" but as a **correlated decision
source**. A serial verifier cascade is then a *fusion system* whose verdicts share blind spots;
the question is how to **allocate verification compute** across correlated sources. This sits
squarely in Information Fusion's stated scope (theoretical analysis, algorithms, multi-classifier
systems, computational-resource optimization), and is not a scope stretch.

**One-line contribution.** *A theory-and-algorithm framework for optimally allocating verification
compute among correlated AI decision sources* — not merely "an analysis of LLM verifier correlation."

## The four decisions (flagship: `fig7_sdse_decision.png`)
At each point a controller chooses among:
- **SCALE** — call the same source again. Useless once its self-correlation saturates
  (same-model 50 calls = **1.2** effective verifiers; usually a trap).
- **DIVERSIFY** — add a complementary source (the only in-pool way to buy independent evidence).
- **STOP** — accept; more verification is below its cost (Theorem 4).
- **ESCALATE** — leave the pool for a stronger/decorrelated source; the *only* way past the
  blind-spot ceiling `1−π₀` (Theorem 2).
Real MAST pools trace the achievable-ceiling curve `C(χ)`: same_model {qwen} is pinned at ceiling
0.14 (χ≈0), rising to 0.86 for the diverse pool (χ≈0.59) — complementarity lifts the ceiling.

## What we add on top of the existing base
The base (item-level estimator, `n_eff`, blind-spot ceiling, submodular selection, probe-driven
guarantee, MAST experiments) already exists. The fusion reframe adds exactly four things:
1. **Budget-aware stopping theorem** (T4) + **safe probe-driven stopping** (T5): when another
   verifier is no longer worth its compute, with a one-sided confidence guarantee.
2. **Scale/Diversify/Stop/Escalate phase diagram** (flagship figure), anchored on real pools.
3. **Verification Debt / Efficiency metric** — `Debt = calls − n_eff`, `Efficiency = n_eff/calls`:
   the compute paid that buys no independent evidence. (same-model qwen: 50 calls, efficiency
   ≈ 0.024; qwen14b repeated: `n_eff@50 ≈ 1.05`.)
4. **Cascade Auditor** (`theory/auditor.py`) — an artifact that ingests a verifier pool (or a small
   probe) and returns `n_eff`, correlation tax, blind-spot ceiling, optimal ordering, `k*`,
   expected reliability, verification debt, and a Scale/Diversify/Stop/Escalate recommendation with
   the warning "adding N more calls is statistically useless."

## Context as a factor: decomposing blindness (the OBSERVE step)

Truncation is not a nuisance to hide — it is a **second decision variable**. We run each verifier
under two context conditions at matched budget: **truncate** (middle removed) vs **evidence-
preserving** (full tail + head + a deterministic structural skeleton of the middle; no LLM
summariser — `cascade/mast_loader.py::_render_evidence_preserving`, selected with
`--context-mode evidence`). Let `β(c)` be the observed blind-spot mass under context condition
`c`. We define

    observed blindness  β(truncate) = β_context + β_intrinsic
    β(evidence-preserving) ≈ β_intrinsic          (the evidence is now visible)
    context-induced blindness  β_context := β(truncate) − β(evidence-preserving)  ≥ 0

and, crucially, the effective independent verifiers `n_eff` is compared across the two conditions.

**Measured outcome (MAST FC3, same-model qwen2.5-7B, seeds held fixed so context is the only
variable; `results/tables/context_decomposition.csv`):**

| mode | n_eff (truncate → evidence) | blind (truncate → evidence) | β_context |
|---|---|---|---|
| FM-3.1 | 1.82 → 1.75 (invariant) | 0.63 → 0.72 | −0.09 |
| FM-3.2 | 1.89 → 1.89 (invariant) | 0.95 → 0.95 | +0.00 |

So on this corpus **β_context ≈ 0 and n_eff is context-invariant**: giving the verifier the full,
evidence-preserving trace does **not** reduce its blindness, and does not change how redundant its
repeats are. The blindness here is **intrinsic (a reasoning failure), not observational** — which
also means the blind-spot magnitudes (0.63, 0.95) are **not truncation artifacts**; the control
locks the claim rather than inflating it. (Most FC3 traces are short — median ≈ 6.2k chars, only
~8–9/40 exceed the 25k budget — so this corpus is not the place to *demonstrate* large
context-induced blindness; it is the place that *rules it out* as a confound.)

*Selection-bias caveat (measured):* the 6 traces too long to fit either context (>131k chars) are
**all ChatDev** with fewer flagged modes (2.3 vs 4.5) — a distinct subpopulation, so they are kept
(as evidence-compressed), never dropped, and this concentration is reported.

The mechanism remains a genuine two-axis story — context governs whether a failure is *seen*,
correlation governs whether repeating a verifier adds *evidence* — but on FC3 the empirical driver
is the correlation axis. The paper's sharpest one-liner still holds as a *limiting* statement:

    More verifier calls cannot recover evidence the verifier cannot see;
    more context cannot remove correlated reasoning blind spots.

So the control loop gains a mandatory first step — **OBSERVE**:

    OBSERVE → DIVERSIFY → SCALE / STOP → ESCALATE

Before spending compute on more verifier calls, the system asks: *is the failure evidence inside
the observable context at all?* If not (`β_context` large), the fix is FIX-CONTEXT (extend /
chunk / retrieve), not more verifiers — pouring 50 calls onto a truncated trace is wasted.

**Rigor guards (Q1).** (i) The evidence-preserving map is a *pure function* of the transcript and
budget — no model-in-the-loop, so no summariser confound. (ii) On this hardware qwen2.5-7B caps at
32 768 tokens (native), so a few extreme transcripts (>131k chars, 4/60) cannot fit either way; we
analyse the **matched non-truncated subset** and run a **selection-bias check** — comparing the
excluded traces vs retained on length, failure subtype, MAS/benchmark — rather than calling it a
"0% dataset." (iii) We do **not** use YaRN/64k: it changes the model's inference regime and would
confound "context extension" with "correlation."

## Honest scope (do not oversell)
The empirical *selection* gap (marginal-greedy vs accuracy-greedy) is only ~1% on real pools,
because real strong verifiers happen to be near-nested (Theorem 6 explains this). We therefore do
**not** claim "our ordering raises accuracy a lot." The magnitude is in **STOP** (same-model
cascades waste ~92% of compute) and **DIVERSIFY/ESCALATE** (ceiling 0.14 → 0.86), and the
contribution is the *framework + guarantees + auditor*, with selection turned into a
characterization theorem rather than a headline win.

## Venue fit
Information Fusion (Q1, resource-optimized fusion of correlated sources) is the primary target.
Fallbacks: TMLR (no-travel Q1), or NeurIPS/ICML/COLM if the empirical breadth (more failure modes,
finite-sample probe on real runs) lands.
