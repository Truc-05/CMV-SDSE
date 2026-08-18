"""MBPP (grade-school-level Python programming) as a verifier error-detection task — the CODE family.

This is a QUALITATIVELY DIFFERENT modality from the MCQA (multiple choice) and GSM8K (numeric
reasoning) families: the object under judgement is a full Python function, and ground truth is
OBJECTIVE — whether the function passes its hidden unit tests when executed. Error items are REAL
buggy programs harvested from an out-of-pool solver (gemma2:9b) and labelled by running the MBPP
`test_list` asserts (see scripts/gen_mbpp_solutions.py). Each cascade item is a problem statement
plus one proposed solution the verifier must judge by READING (it never executes anything):

  - error item     = a solution that FAILS at least one unit test  (ground_truth_is_error=True)
  - non-error item = a solution that PASSES every unit test         (ground_truth_is_error=False)

`accepted=True` (NO_FAILURE) on an error item is a false-accept (miss); `accepted=False`
(FAILURE) on a clean item is a false alarm — identical polarity to the MAST / MCQA / GSM8K
loaders, so the whole downstream (Beta-Binomial, marginal value, phase, n_eff) is reused
unchanged. To keep the same generator!=verifier separation and the same "no oracle shown to the
verifier" rule as GSM8K, the SOLVER sees the tests (to fix the function signature) but the
VERIFIER sees only the problem statement and the code — never the unit tests.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from cascade.ollama_backend import VERDICT_INSTRUCTION
from cascade.scenario_loader import CascadeItem

CACHE = Path(__file__).resolve().parents[1] / "data" / "processed" / "mbpp_solutions.jsonl"
FM = "MBPP"


def mbpp_system_prompt() -> str:
    """FAILURE == the proposed program is incorrect (fails its unit tests == ground_truth_is_error)."""
    return (
        "You are checking whether a proposed Python function correctly solves a described "
        "programming task.\n\n"
        "You will be given a task description and a single proposed solution (one Python "
        "function).\n\n"
        "Decide one thing only: Is the proposed solution incorrect, that is, would it fail to "
        "correctly implement the described task on valid inputs (a bug, wrong logic, wrong "
        "return value, a crash, or a wrong edge case).\n\n"
        "Reason about the code yourself if needed: trace it on plausible inputs. Judge only "
        "functional correctness, not style, naming, or efficiency.\n\n"
        + VERDICT_INSTRUCTION
    )


def _render(text: str, code: str) -> str:
    return (f"Task:\n{str(text).strip()}\n\n"
            f"Proposed Python solution:\n{str(code).strip()}")


def _load_cache():
    if not CACHE.exists():
        raise FileNotFoundError(
            f"{CACHE} not found — run scripts/gen_mbpp_solutions.py first to harvest and "
            "label gemma2 MBPP solutions.")
    recs = []
    for line in open(CACHE, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("parsed") and r.get("code"):
                recs.append(r)
    return recs


def load_mbpp_items(n_errors: int, n_non_errors: int,
                    seed: int = 20260701, **_ignored) -> list[CascadeItem]:
    recs = _load_cache()
    errors = [r for r in recs if r["is_error"]]
    cleans = [r for r in recs if not r["is_error"]]
    if len(errors) < n_errors or len(cleans) < n_non_errors:
        raise ValueError(
            f"mbpp cache has {len(errors)} error / {len(cleans)} clean solutions; "
            f"need {n_errors}/{n_non_errors}. Generate more with gen_mbpp_solutions.py.")

    rng = random.Random(seed)
    rng.shuffle(errors)
    rng.shuffle(cleans)

    items = []
    for r in errors[:n_errors]:
        items.append(CascadeItem(
            item_id=f"{FM}#{r['split']}_{r['task_id']}#err", ground_truth_is_error=True,
            payload={"prompt": _render(r["text"], r["code"])},
            scenario_id=FM, failure_mode_id=FM, source_path="google-research-datasets/mbpp",
            source_config={"solver": "gemma2:9b", "task_id": r["task_id"],
                           "n_tests": r.get("n_tests"), "n_passed": r.get("n_passed")}))
    for r in cleans[:n_non_errors]:
        items.append(CascadeItem(
            item_id=f"{FM}#{r['split']}_{r['task_id']}#ok", ground_truth_is_error=False,
            payload={"prompt": _render(r["text"], r["code"])},
            scenario_id=FM, failure_mode_id=FM, source_path="google-research-datasets/mbpp",
            source_config={"solver": "gemma2:9b", "task_id": r["task_id"],
                           "n_tests": r.get("n_tests"), "n_passed": r.get("n_passed")}))
    rng.shuffle(items)
    return items
