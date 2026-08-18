# Budget-Aware Fusion of Correlated LLM Verifiers
### *When more verifiers are not more information — when to Scale, Diversify, Stop, or Escalate*

Stacking LLM verifiers assumes more calls give more evidence. They don't when the calls are
**correlated**. This repo measures that gap and turns it into an acquisition policy.

**Findings (7 task families: MAST, ARC, MMLU, CSQA, TruthfulQA, GSM8K, MBPP)**
- **Redundancy.** 50 same-model calls ≈ **1.06–1.21** effective independent verifiers (`n_eff@50`)
  on the ≤14B pool (1.44–2.29 at 24B) — nominal effort ×50, effective evidence barely moves.
- **Diversity ≠ decision value.** A statistically diverse verifier can catch *zero* residual errors;
  the quantity that matters is **conditional marginal value** (residual catch, priced against false
  alarms and compute).
- **CMV-SDSE.** A budget-aware controller that **S**cales / **D**iversifies / **S**tops / **E**scalates
  on that value, and in its joint form co-selects the **fusion topology** (OR↔AND↔…) with the source set.
- One value-selected verifier reaches **U=0.486** vs 0.233 for 50 same-model calls (~**117×** less compute).

Paper (Information Fusion submission): [`nd.md`](nd.md).

## Repo layout
| Path | What |
|---|---|
| `run_cascade.py` | main runner — collects verifier verdicts into `data/raw_cascade_runs/` |
| `cascade/` | task loaders (`mcqa/gsm8k/mbpp/mast_loader`), `gate_runner`, Ollama backend, verifier chain |
| `theory/` | `marginal_value` (CMV, `n_eff`, `_prep`), Beta-Binomial fit, ceiling/stopping tests |
| `scripts/` | offline analysis (`reviewer_analyses`, `fusion_selection_baselines`, `joint_topology_controller`, …) + figure builders (`make_figures3`, `new_fig_real`) |
| `results/tables/` | all result CSVs (every paper number reconciles to one) |
| `data/` | `raw_cascade_runs/` (verdicts) · `processed/` (generated GSM8K/MBPP solver solutions) |
| `figures/` · `figure3/` · `figure4/` | paper figures (PDF/PNG) |
| `configs/` | `models.yaml` (verifiers, decoding), `cascade_thresholds.yaml` (frozen, SHA-pinned) |
| `docs/` | theory notes, `METHODOLOGY_FREEZE.md` |

## Install & run
Interpreter: `~/venvs/env/bin/python`. Data collection needs a local **Ollama** server hosting the
verifiers (`qwen2.5:7b`, `llama3.1:8b`, `llama3.2:3b`, `mistral:7b`; generator `gemma2:9b`) — set the host
in `configs/models.yaml`.

```bash
source ~/venvs/env/bin/activate
pip install numpy pandas matplotlib pyyaml requests datasets

# 1) collect verdicts (GPU/Ollama).  --source: mast | mcqa | gsm8k | mbpp
python run_cascade.py --source mcqa --benchmark csqa --pool same_model \
       --k 50 --n-errors 150 --n-non-errors 75 --context-mode truncate --skip-analyse

# 2) offline analysis (no GPU — reads the frozen verdict CSVs)
python scripts/reviewer_analyses.py          # acquisition ablation, budget curve, regret, probes
python scripts/fusion_selection_baselines.py # CMV-SDSE vs GES / mRMR (in-sample + held-out)
python scripts/joint_topology_controller.py  # joint source–topology controller (held-out)

# 3) figures
python scripts/make_figures3.py              # main paper figures -> figure3/
python scripts/new_fig_real.py               # recovery / generalization / joint-policy -> figure4/
```

## Notes
- **Reproducibility:** the controller, metric, and hyperparameters were frozen before the confirmatory
  runs (`cascade_thresholds.yaml`, SHA-pinned); offline scripts re-run byte-identically from the released
  verdicts. `theory.marginal_value._prep` excludes out-of-pool probes (24B, gemma) so frozen tables stand.
- **Scope (no overclaim):** results are stated for the *seven evaluated families* and ≤14B core pool
  (+24B escalation); `π₀` is reported as an upper bound only; no universal fusion rule is claimed.
