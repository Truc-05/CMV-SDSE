# Methodology freeze — 2026-08-09

Status: **FROZEN.** The controller and its hyperparameters are locked as of this date.
No further changes to controller logic or tuning. Next step is confirmatory large-n
(the numbers below are the exploratory/replication phase; large-n re-runs them at scale
without re-tuning).

## Locked hyperparameters
| knob | frozen value | where |
|---|---|---|
| canonical population | errors-only, `n_gates == 50` | all analyses |
| decision context | `ctx_truncate` (primary) | raw run dirs `*__ctx_truncate` |
| interpreter | `~/venvs/env/bin/python` (`.venv` is broken) | — |
| classifier threshold | `0.5` | `_prep(df, 0.5)` |
| greedy cost weight | `lam = 0.1` | `greedy_marginal_value(df, lam=0.1)` |
| add rule (invariant) | diversify/scale a candidate **only** when Wilson CI-low(Δcatch) > 0 | `theory/marginal_value.py` |
| redundancy estimator | Beta-Binomial mixture, `n_eff_at_k_max` @ k=50 | `theory/betabinom_mixture.fit_mixture` |
| verifier universe | `llama31, llama32_3b, mistral7b, qwen7b` | per family |
| phase grid | fa/miss ∈ {0,.25,.5,1,2,4}, λ ∈ {0,.05,.1,.2,.4,.8} | `phase_diagram` |

## What is claimed (disciplined wording)
- **Redundancy claim.** Repeated same-model gates add almost no independent evidence:
  same-model `n_eff@50 ∈ [1.02, 1.21]` **replicated across five evaluated benchmark/task
  families** (MAST-3.3, ARC, MMLU, CSQA, TruthfulQA). See
  `results/tables/redundancy_invariant_5family.csv`.
- **Controller invariant.** No fixed model-identity or family label decides which verifier
  should be added; the controller must measure **conditional marginal catch on the current
  task**. Across the 4 MCQA families the invariant holds (add ⇔ Wilson CI-low > 0), the
  two weak verifiers (`llama31`, `llama32_3b`) are correctly never diversified in, and
  exactly one diverse verifier is added before STOP.

### Wording guardrails — do NOT write:
- ❌ "works on any benchmark / any model"
- ✅ "replicated across five evaluated benchmark/task families"
- The positive diversify signal is **underpowered** (pooled 9/15 residual catches;
  MCQA is near-ceiling — solo verifier already catches 85–97.5%). State it as directional
  and let confirmatory large-n do the powering. The well-powered result is the redundancy
  null (40 items × 50 gates/family).

## Benign artifact (documented, not a bug)
Phase-diagram `S` (SCALE) cells for TruthfulQA appear only when `s0 ∈ {llama31,
llama32_3b}` with `set_size==1` — i.e. a degenerate seed starting from a dead verifier
under forced action. Not a genuine recommendation to scale; consistent with "nothing real
to gain." No controller change.

## Next (confirmatory)
Large-n re-run of the same pipeline & frozen hyperparameters. No re-tuning.
