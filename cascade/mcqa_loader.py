"""Multiple-choice QA benchmarks as a verifier error-detection task (non-MAST replication).

Each question yields items the cascade must judge for correctness:
  - error item     = question + a WRONG option   (ground_truth_is_error=True  -> should be caught)
  - non-error item = question + the CORRECT option (ground_truth_is_error=False -> should pass)

`accepted=True` (NO_FAILURE) on an error item is a false-accept (miss); `accepted=False`
(FAILURE) on a clean item is a false alarm — identical polarity to the MAST loader, so the
whole downstream (Beta-Binomial, marginal value, phase, n_eff) is reused unchanged. Only this
loader and the verifier system prompt differ from the MAST path.

Benchmarks (all cached locally): arc, mmlu, csqa, truthfulqa.
"""
from __future__ import annotations

import random

from cascade.scenario_loader import CascadeItem


# ---- per-benchmark row -> (question, correct_answer, [wrong_answers]) --------------------
def _arc_style(row):
    """ai2_arc / commonsense_qa: choices={'text':[...], 'label':[...]}, answerKey=label."""
    ch = row["choices"]
    labels, texts = ch["label"], ch["text"]
    key = row.get("answerKey")
    if key not in labels:
        return None
    ci = labels.index(key)
    return row["question"], texts[ci], [t for i, t in enumerate(texts) if i != ci]


def _mmlu(row):
    """cais/mmlu: choices=[4 strings], answer=int index."""
    ch, ans = row["choices"], row["answer"]
    if not isinstance(ans, int) or not (0 <= ans < len(ch)):
        return None
    return row["question"], ch[ans], [c for i, c in enumerate(ch) if i != ans]


def _truthfulqa(row):
    """truthful_qa (multiple_choice): mc1_targets={'choices':[...], 'labels':[0/1]}."""
    mc = row["mc1_targets"]
    choices, labels = mc["choices"], mc["labels"]
    if 1 not in labels:
        return None
    ci = labels.index(1)
    return row["question"], choices[ci], [c for i, c in enumerate(choices) if labels[i] == 0]


# name -> (hf_path, hf_config, split, extractor, failure_mode_label)
BENCHMARKS = {
    "arc":        ("allenai/ai2_arc", "ARC-Challenge", "validation", _arc_style, "ARC"),
    "mmlu":       ("cais/mmlu", "all", "validation", _mmlu, "MMLU"),
    "csqa":       ("tau/commonsense_qa", None, "validation", _arc_style, "CSQA"),
    "truthfulqa": ("truthfulqa/truthful_qa", "multiple_choice", "validation", _truthfulqa, "TRUTHFULQA"),
}


def mcqa_system_prompt() -> str:
    """FAILURE == the proposed answer is wrong (== ground_truth_is_error)."""
    return (
        "You are checking whether a proposed answer to a question is correct.\n\n"
        "You will be given a question and a single proposed answer.\n\n"
        "Decide one thing only: Is the proposed answer incorrect, that is, a wrong answer to "
        "the question.\n\n"
        "Judge only the factual correctness of the proposed answer. Do not grade style or "
        "phrasing.\n\n"
        "Your entire reply must be exactly one line, and that line must be either\n"
        "VERDICT: FAILURE\n"
        "or\n"
        "VERDICT: NO_FAILURE\n"
        "Do not explain. Do not restate the question. Do not write anything before or after "
        "that line."
    )


def _render(question: str, answer: str) -> str:
    return f"Question:\n{str(question).strip()}\n\nProposed answer:\n{str(answer).strip()}"


def load_mcqa_items(benchmark: str, n_errors: int, n_non_errors: int,
                    seed: int = 20260701, **_ignored) -> list[CascadeItem]:
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown mcqa benchmark {benchmark!r}; have {sorted(BENCHMARKS)}")
    from datasets import load_dataset

    hf_path, hf_config, split, extractor, fm = BENCHMARKS[benchmark]
    ds = load_dataset(hf_path, hf_config, split=split)

    rows = []
    for r in ds:
        got = extractor(r)
        if not got:
            continue
        q, correct, wrongs = got
        if q and correct and wrongs:
            rows.append((q, correct, wrongs))

    rng = random.Random(seed)
    rng.shuffle(rows)
    need = n_errors + n_non_errors
    if len(rows) < need:
        raise ValueError(f"{benchmark}: only {len(rows)} usable questions, need {need}")

    items = []
    for idx, (q, _correct, wrongs) in enumerate(rows[:n_errors]):
        items.append(CascadeItem(
            item_id=f"{fm}#{idx}#err", ground_truth_is_error=True,
            payload={"prompt": _render(q, rng.choice(wrongs))},
            scenario_id=fm, failure_mode_id=fm, source_path=hf_path, source_config={}))
    for idx, (q, correct, _wrongs) in enumerate(rows[n_errors:need]):
        items.append(CascadeItem(
            item_id=f"{fm}#{idx}#ok", ground_truth_is_error=False,
            payload={"prompt": _render(q, correct)},
            scenario_id=fm, failure_mode_id=fm, source_path=hf_path, source_config={}))
    rng.shuffle(items)
    return items
