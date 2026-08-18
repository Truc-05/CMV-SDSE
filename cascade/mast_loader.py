from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Iterator

from cascade.scenario_loader import CascadeItem

REPO_CANDIDATES = ("mcemri/MAD", "mcemri/MAST-Data")
FULL_FILENAME = "MAD_full_dataset.json"
HUMAN_FILENAME = "MAD_human_labelled_dataset.json"

MODE_IDS = ["1.1", "1.2", "1.3", "1.4", "1.5",
            "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
            "3.1", "3.2", "3.3"]

MODE_NAMES = {
    "1.1": "Disobey task specification",
    "1.2": "Disobey role specification",
    "1.3": "Step repetition",
    "1.4": "Loss of conversation history",
    "1.5": "Unaware of stopping conditions",
    "2.1": "Conversation reset",
    "2.2": "Fail to ask for clarification",
    "2.3": "Task derailment",
    "2.4": "Information withholding",
    "2.5": "Ignored other agent input",
    "2.6": "Reasoning-action mismatch",
    "3.1": "Premature termination",
    "3.2": "No or incomplete verification",
    "3.3": "Incorrect verification",
}

FC3_MODES = ["3.1", "3.2", "3.3"]

MODE_QUESTIONS = {
    "1.1": "did any agent fail to honour a stated constraint or requirement of the task",
    "1.2": "did any agent act outside the responsibilities of its assigned role",
    "1.3": "did any agent repeat a step it had already completed",
    "1.4": "did any agent lose or contradict earlier content of the conversation",
    "1.5": "did the exchange continue after the task was already complete",
    "2.1": "was the dialogue restarted or reset unexpectedly",
    "2.2": "did an agent proceed on unclear information instead of asking for clarification",
    "2.3": "did the exchange drift away from the assigned objective",
    "2.4": "did an agent hold back information that another agent needed",
    "2.5": "did an agent disregard input or a correction from another agent",
    "2.6": "did any agent take an action that contradicted its own stated reasoning",
    "3.1": "did the exchange end before the task was finished",
    "3.2": "was the output left unverified, or verified only superficially",
    "3.3": "was a verification performed but reached the wrong conclusion",
}

FC3_QUESTION = (
    "did the system fail at verification, meaning it ended before the task was finished, "
    "left its output unverified or only superficially checked, or performed a verification "
    "that reached the wrong conclusion"
)

ANY_QUESTION = (
    "did a failure occur during this exchange, where a failure means the agents did not "
    "accomplish what the task required of them, for any reason"
)

FULL_REQUIRED = {"mas_name", "benchmark_name", "trace_id", "trace", "mast_annotation"}
HUMAN_REQUIRED = {"mas_name", "benchmark_name", "trace_id", "trace", "annotations"}

MODE_ID_PATTERN = re.compile(r"^\s*(\d+\.\d+)")


class MastSchemaError(RuntimeError):
    pass


class MastDownloadError(RuntimeError):
    pass


def question_for(target: str) -> str:
    if target == "any":
        return ANY_QUESTION
    if target == "fc3":
        return FC3_QUESTION
    if target in MODE_QUESTIONS:
        return MODE_QUESTIONS[target]
    raise ValueError(f"no verifier question defined for target {target!r}")


def system_prompt_for(target: str) -> str:
    return (
        "You are reviewing a transcript of an exchange between automated agents working on a "
        "task.\n\nYou will be given the task brief and the full transcript.\n\n"
        f"Decide one thing only: {question_for(target)}.\n\n"
        "Do not speculate about intent. Do not grade style, tone, or efficiency. Judge only "
        "what the transcript shows.\n\n"
        "Your entire reply must be exactly one line, and that line must be either\n"
        "VERDICT: FAILURE\n"
        "or\n"
        "VERDICT: NO_FAILURE\n"
        "Do not explain. Do not restate the question. Do not write anything before or "
        "after that line."
    )


def _download(filename: str, cache_dir: str | None = None) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise MastDownloadError(
            "huggingface_hub is not installed. Run: pip install huggingface_hub"
        ) from exc

    errors = []
    for repo in REPO_CANDIDATES:
        try:
            return hf_hub_download(repo_id=repo, filename=filename,
                                   repo_type="dataset", cache_dir=cache_dir)
        except Exception as exc:
            errors.append(f"{repo}: {type(exc).__name__}: {str(exc)[:160]}")
    raise MastDownloadError(
        "could not download " + filename + " from any known repo.\n  "
        + "\n  ".join(errors)
        + "\nIf the machine has no network access to huggingface.co, download the file "
          "manually and pass --mast-path <dir>."
    )


def resolve_path(filename: str, local_dir: str | None = None,
                 cache_dir: str | None = None) -> str:
    if local_dir:
        candidate = os.path.join(local_dir, filename)
        if not os.path.exists(candidate):
            raise MastDownloadError(f"{candidate} not found")
        return candidate
    return _download(filename, cache_dir)


def _load_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise MastSchemaError(f"{path} did not contain a non-empty list of records")
    return data


def validate_full(records: list[dict[str, Any]]) -> None:
    keys = set(records[0])
    missing = FULL_REQUIRED - keys
    if missing:
        raise MastSchemaError(
            f"full dataset is missing expected fields {sorted(missing)}; "
            f"found {sorted(keys)}. The upstream schema may have changed."
        )
    ann = records[0]["mast_annotation"]
    if not isinstance(ann, dict):
        raise MastSchemaError("mast_annotation is not a dict")
    unknown = set(ann) - set(MODE_IDS)
    if unknown:
        raise MastSchemaError(f"unexpected failure mode keys: {sorted(unknown)}")


def validate_human(records: list[dict[str, Any]]) -> None:
    keys = set(records[0])
    missing = HUMAN_REQUIRED - keys
    if missing:
        raise MastSchemaError(
            f"human dataset is missing expected fields {sorted(missing)}; "
            f"found {sorted(keys)}"
        )
    ann = records[0]["annotations"]
    if not isinstance(ann, list) or not ann:
        raise MastSchemaError("annotations is not a non-empty list")
    entry_keys = set(ann[0])
    if not {"annotator_1", "annotator_2", "annotator_3"} <= entry_keys:
        raise MastSchemaError(f"annotation entry keys unexpected: {sorted(entry_keys)}")


def trajectory_of(record: dict[str, Any]) -> str:
    trace = record.get("trace")
    if isinstance(trace, dict):
        return str(trace.get("trajectory", ""))
    return str(trace or "")


def is_error_for(annotation: dict[str, Any], target: str) -> bool:
    if target == "any":
        return any(int(annotation.get(m) or 0) == 1 for m in MODE_IDS)
    if target == "fc3":
        return any(int(annotation.get(m) or 0) == 1 for m in FC3_MODES)
    if target not in MODE_IDS:
        raise ValueError(f"unknown target mode {target!r}; use any, fc3, or one of {MODE_IDS}")
    return int(annotation.get(target) or 0) == 1


def _render_evidence_preserving(trajectory: str, max_chars: int) -> str:
    """Deterministic evidence-preserving compression to fit `max_chars`.

    Keeps the full HEAD (task setup) and a large full TAIL (where FC3 / task-verification
    evidence concentrates), and replaces the MIDDLE with a structural SKELETON — the first
    ~90 chars of each middle line — so a *missing* verification step stays visible without the
    full text. No LLM is involved (that would add a summariser confound); the mapping is a pure
    function of the transcript and the budget.
    """
    head_budget = int(0.15 * max_chars)
    tail_budget = int(0.65 * max_chars)
    skel_budget = max_chars - head_budget - tail_budget
    head = trajectory[:head_budget]
    tail = trajectory[len(trajectory) - tail_budget:]
    middle = trajectory[head_budget: len(trajectory) - tail_budget]

    lines = [ln.strip() for ln in middle.splitlines() if ln.strip()]
    max_lines = max(skel_budget // 91, 1)
    # stride-sample evenly across the WHOLE middle so a missing step anywhere stays visible
    if len(lines) > max_lines:
        stride = len(lines) / max_lines
        idx = sorted({int(i * stride) for i in range(max_lines)})
        lines = [lines[j] for j in idx if j < len(lines)]
    kept, used = [], 0
    for line in lines:
        snippet = line[:90]
        if used + len(snippet) + 1 > skel_budget:
            break
        kept.append(snippet)
        used += len(snippet) + 1
    n_lines = len(kept)
    skeleton = "\n".join(kept)
    dropped = len(middle)
    return (
        head
        + f"\n\n[... structural skeleton of {dropped} middle characters "
          f"({n_lines} step/line openers kept; full text of the middle removed) ...]\n"
        + skeleton
        + "\n[... end of skeleton; the full ending of the transcript follows intact ...]\n\n"
        + tail
    )


def render_prompt(record: dict[str, Any], max_chars: int | None = None,
                  context_mode: str = "truncate") -> str:
    trajectory = trajectory_of(record)
    if max_chars and len(trajectory) > max_chars:
        if context_mode == "evidence":
            trajectory = _render_evidence_preserving(trajectory, max_chars)
        else:
            dropped = len(trajectory) - max_chars
            head = trajectory[: max_chars // 2]
            tail = trajectory[-max_chars // 2:]
            trajectory = (
                head
                + f"\n\n[... {dropped} characters of the middle of this transcript were "
                  "removed to fit the context window; the opening and the ending are intact ...]\n\n"
                + tail
            )
    return (
        "Task brief.\n"
        f"A multi-agent system named {record.get('mas_name', 'unknown')} was run on the "
        f"{record.get('benchmark_name', 'unknown')} benchmark. It was required to complete the "
        "task the benchmark sets.\n\n"
        f"Transcript.\n{trajectory}"
    )


def iter_mast_items(full_path: str, target: str = "any",
                    mas_names: list[str] | None = None,
                    benchmarks: list[str] | None = None,
                    max_chars: int | None = None,
                    context_mode: str = "truncate") -> Iterator[CascadeItem]:
    records = _load_json(full_path)
    validate_full(records)
    for record in records:
        if mas_names and record.get("mas_name") not in mas_names:
            continue
        if benchmarks and record.get("benchmark_name") not in benchmarks:
            continue
        trajectory = trajectory_of(record)
        if not trajectory.strip():
            continue
        annotation = record["mast_annotation"]
        yield CascadeItem(
            item_id=(f"{record.get('mas_name')}|{record.get('benchmark_name')}|"
                     f"{record.get('llm_name', 'na')}|{record.get('trace_id')}"),
            ground_truth_is_error=is_error_for(annotation, target),
            payload={"prompt": render_prompt(record, max_chars, context_mode)},
            scenario_id=f"mast:{target}",
            failure_mode_id=("ANY" if target == "any"
                             else "FC3" if target == "fc3" else f"FM-{target}"),
            source_path=full_path,
            source_config={
                "mas_name": record.get("mas_name"),
                "llm_name": record.get("llm_name"),
                "benchmark_name": record.get("benchmark_name"),
                "trace_chars": len(trajectory),
                "modes_flagged": [m for m in MODE_IDS if int(annotation.get(m) or 0) == 1],
                "label_source": "mast_llm_annotator",
                "context_mode": context_mode,
                "raw_exceeds_context": bool(max_chars and len(trajectory) > max_chars),
                # destructive middle-cut only in truncate mode; evidence mode keeps a full-trace
                # skeleton + full tail, so it is compressed, NOT information-destructively truncated
                "was_truncated": bool(max_chars and len(trajectory) > max_chars
                                      and context_mode != "evidence"),
                "was_compressed": bool(max_chars and len(trajectory) > max_chars
                                       and context_mode == "evidence"),
            },
        )


def load_mast_items(full_path: str, target: str = "any",
                    n_errors: int | None = None, n_non_errors: int | None = None,
                    mas_names: list[str] | None = None,
                    benchmarks: list[str] | None = None,
                    max_chars: int | None = None,
                    context_mode: str = "truncate",
                    shuffle_seed: int = 20260701) -> list[CascadeItem]:
    import random

    items = list(iter_mast_items(full_path, target, mas_names, benchmarks, max_chars, context_mode))
    errors = [i for i in items if i.ground_truth_is_error]
    clean = [i for i in items if not i.ground_truth_is_error]

    rng = random.Random(shuffle_seed)
    rng.shuffle(errors)
    rng.shuffle(clean)
    if n_errors is not None:
        errors = errors[:n_errors]
    if n_non_errors is not None:
        clean = clean[:n_non_errors]

    selected = errors + clean
    rng.shuffle(selected)
    return selected


def describe(full_path: str, target: str = "any") -> dict[str, Any]:
    records = _load_json(full_path)
    validate_full(records)
    mas = Counter(r.get("mas_name") for r in records)
    bench = Counter(r.get("benchmark_name") for r in records)
    llm = Counter(r.get("llm_name") for r in records)
    per_mode = {m: sum(int(r["mast_annotation"].get(m, 0)) for r in records) for m in MODE_IDS}
    n_error = sum(1 for r in records if is_error_for(r["mast_annotation"], target))
    total_flags = sum(per_mode.values())
    return {
        "n_records": len(records),
        "target": target,
        "n_error": n_error,
        "n_clean": len(records) - n_error,
        "total_mode_flags": total_flags,
        "flags_per_trace": round(total_flags / max(len(records), 1), 3),
        "by_mas": dict(mas),
        "by_benchmark": dict(bench),
        "by_llm": dict(llm),
        "per_mode_counts": per_mode,
        "per_mode_prevalence": {m: round(c / max(len(records), 1), 4)
                                for m, c in per_mode.items()},
        "per_mode_share_of_flags": {m: round(c / max(total_flags, 1), 4)
                                    for m, c in per_mode.items()},
    }


def normalise_mode_label(raw: str) -> tuple[str, str]:
    text = " ".join(str(raw).split())
    match = MODE_ID_PATTERN.match(text)
    mode_id = match.group(1) if match else "unparsed"
    title = text[match.end():].strip() if match else text
    return mode_id, title[:60]


@dataclass
class LabelNoiseReport:
    n_traces: int
    n_annotations: int
    rounds: dict[str, int]
    mode_ids_seen: list[str]
    ids_outside_final_taxonomy: list[str]
    ids_with_multiple_titles: dict[str, list[str]]
    per_mode: dict[str, dict[str, float]]
    overall_full_agreement: float
    overall_full_agreement_informative: float
    n_informative: int
    overall_majority_positive_rate: float
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyse_label_noise(human_path: str) -> LabelNoiseReport:
    records = _load_json(human_path)
    validate_human(records)

    per_mode: dict[str, dict[str, float]] = {}
    titles: dict[str, set] = {}
    rounds: Counter = Counter()
    total_entries = 0
    full_agreement = 0
    informative = 0
    informative_unanimous = 0
    majority_positive = 0

    for record in records:
        rounds[str(record.get("round", "unknown"))] += 1
        for entry in record["annotations"]:
            raw = entry.get("failure mode", entry.get("failure_mode", ""))
            mode_id, title = normalise_mode_label(raw)
            titles.setdefault(mode_id, set()).add(title)
            votes = [bool(entry.get(f"annotator_{i}", False)) for i in (1, 2, 3)]
            row = per_mode.setdefault(mode_id, {
                "n": 0, "unanimous": 0, "majority_positive": 0, "any_positive": 0,
                "n_informative": 0, "informative_unanimous": 0,
            })
            row["n"] += 1
            total_entries += 1
            unanimous = len(set(votes)) == 1
            if unanimous:
                row["unanimous"] += 1
                full_agreement += 1
            if any(votes):
                row["any_positive"] += 1
                row["n_informative"] += 1
                informative += 1
                if unanimous:
                    row["informative_unanimous"] += 1
                    informative_unanimous += 1
            if sum(votes) >= 2:
                row["majority_positive"] += 1
                majority_positive += 1

    for mode_id, row in per_mode.items():
        n = max(row["n"], 1)
        ni = max(row["n_informative"], 1)
        row["unanimous_rate"] = row["unanimous"] / n
        row["majority_positive_rate"] = row["majority_positive"] / n
        row["any_positive_rate"] = row["any_positive"] / n
        row["disagreement_rate"] = 1.0 - row["unanimous_rate"]
        row["informative_unanimous_rate"] = (
            row["informative_unanimous"] / ni if row["n_informative"] else float("nan")
        )

    outside = sorted(m for m in per_mode if m not in MODE_IDS and m != "unparsed")
    multi = {m: sorted(t) for m, t in titles.items() if len(t) > 1}

    return LabelNoiseReport(
        n_traces=len(records),
        n_annotations=total_entries,
        rounds=dict(rounds),
        mode_ids_seen=sorted(per_mode),
        ids_outside_final_taxonomy=outside,
        ids_with_multiple_titles=multi,
        per_mode=per_mode,
        overall_full_agreement=full_agreement / max(total_entries, 1),
        overall_full_agreement_informative=(
            informative_unanimous / informative if informative else float("nan")
        ),
        n_informative=informative,
        overall_majority_positive_rate=majority_positive / max(total_entries, 1),
        note=(
            "Two cautions. First, unanimity computed over every row is inflated: most rows are "
            "unanimous negatives, which are trivial to agree on. The informative rate, computed "
            "only over rows where at least one annotator said yes, is the number that matters. "
            "Second, the released human file spans several inter-annotator rounds that used "
            "different draft taxonomies, so the same numeric id can carry different definitions "
            "across rounds and some ids do not appear in the final fourteen-mode taxonomy at "
            "all. Do not pool across rounds without checking ids_with_multiple_titles and "
            "ids_outside_final_taxonomy."
        ),
    )
