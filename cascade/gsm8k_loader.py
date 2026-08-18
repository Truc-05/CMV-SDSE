"""GSM8K (grade-school math) as a verifier error-detection task — the reasoning family.

Unlike the MCQA benchmarks, GSM8K has no ready-made distractors, so error items are REAL
multi-step reasoning failures harvested from an out-of-pool solver (gemma2:9b) and labelled
against the gold answer (see scripts/gen_gsm8k_solutions.py). Each cascade item is a full
worked solution the verifier must judge:

  - error item     = a solution whose final answer is WRONG   (ground_truth_is_error=True)
  - non-error item = a solution whose final answer is CORRECT (ground_truth_is_error=False)

`accepted=True` (NO_FAILURE) on an error item is a false-accept (miss); `accepted=False`
(FAILURE) on a clean item is a false alarm — identical polarity to the MAST / MCQA loaders,
so the whole downstream (Beta-Binomial, marginal value, phase, n_eff) is reused unchanged.
Only this loader and the verifier system prompt differ from the MCQA path.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from cascade.ollama_backend import VERDICT_INSTRUCTION
from cascade.scenario_loader import CascadeItem

CACHE = Path(__file__).resolve().parents[1] / "data" / "processed" / "gsm8k_solutions.jsonl"
FM = "GSM8K"


def gsm8k_system_prompt() -> str:
    """FAILURE == the proposed solution's final answer is wrong (== ground_truth_is_error)."""
    return (
        "You are checking whether a proposed step-by-step solution to a grade-school math "
        "word problem reaches the correct final answer.\n\n"
        "You will be given a question and a single proposed solution with its final answer.\n\n"
        "Decide one thing only: Is the proposed solution incorrect, that is, does it reach a "
        "wrong final answer to the question.\n\n"
        "Work the problem yourself if needed. Judge only the correctness of the final answer, "
        "not the style or wording of the solution.\n\n"
        + VERDICT_INSTRUCTION
    )


def _render(question: str, solution: str) -> str:
    return (f"Question:\n{str(question).strip()}\n\n"
            f"Proposed solution:\n{str(solution).strip()}")


def _load_cache():
    if not CACHE.exists():
        raise FileNotFoundError(
            f"{CACHE} not found — run scripts/gen_gsm8k_solutions.py first to harvest and "
            "label gemma2 GSM8K solutions.")
    recs = []
    for line in open(CACHE, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("parsed") and r.get("solution"):
                recs.append(r)
    return recs


def load_gsm8k_items(n_errors: int, n_non_errors: int,
                     seed: int = 20260701, **_ignored) -> list[CascadeItem]:
    recs = _load_cache()
    errors = [r for r in recs if r["is_error"]]
    cleans = [r for r in recs if not r["is_error"]]
    if len(errors) < n_errors or len(cleans) < n_non_errors:
        raise ValueError(
            f"gsm8k cache has {len(errors)} error / {len(cleans)} clean solutions; "
            f"need {n_errors}/{n_non_errors}. Generate more with gen_gsm8k_solutions.py.")

    rng = random.Random(seed)
    rng.shuffle(errors)
    rng.shuffle(cleans)

    items = []
    for i, r in enumerate(errors[:n_errors]):
        items.append(CascadeItem(
            item_id=f"{FM}#{r['split']}_{r['idx']}#err", ground_truth_is_error=True,
            payload={"prompt": _render(r["question"], r["solution"])},
            scenario_id=FM, failure_mode_id=FM, source_path="openai/gsm8k",
            source_config={"solver": "gemma2:9b", "gold": r["gold"], "pred": r["pred"]}))
    for i, r in enumerate(cleans[:n_non_errors]):
        items.append(CascadeItem(
            item_id=f"{FM}#{r['split']}_{r['idx']}#ok", ground_truth_is_error=False,
            payload={"prompt": _render(r["question"], r["solution"])},
            scenario_id=FM, failure_mode_id=FM, source_path="openai/gsm8k",
            source_config={"solver": "gemma2:9b", "gold": r["gold"], "pred": r["pred"]}))
    rng.shuffle(items)
    return items
