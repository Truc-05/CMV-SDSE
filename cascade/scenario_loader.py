from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Iterator

TASK_BRIEFS = {
    "fm_2_4": "File the shipment manifest carrying the correct dispatch code.",
    "fm_2_5": "Issue a price quote that uses the current unit rate.",
    "fm_1_3": "Validate every invoice in the batch exactly once.",
    "fm_1_4": "Answer the questions, then report the project codename from the briefing.",
    "fm_1_5": "Run every required release check, then close the release.",
    "fm_3_1": "Run every required release check, then close the release.",
    "fm_1_1": (
        "Implement daily_word(dictionary_path) returning a random word, reading candidates "
        "from the supplied path rather than a hard-coded collection."
    ),
}

FAILURE_MODE_BY_SCENARIO = {
    "fm_1_1": "FM-1.1", "fm_1_3": "FM-1.3", "fm_1_4": "FM-1.4", "fm_1_5": "FM-1.5",
    "fm_2_4": "FM-2.4", "fm_2_5": "FM-2.5", "fm_3_1": "FM-3.1",
}


@dataclass
class CascadeItem:
    item_id: str
    ground_truth_is_error: bool
    payload: dict[str, Any]
    scenario_id: str
    failure_mode_id: str
    source_path: str
    source_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def render_prompt(trace: dict[str, Any]) -> str:
    scenario_id = trace.get("scenario_id", "")
    brief = TASK_BRIEFS.get(scenario_id, "Complete the assigned task.")
    lines = [
        f"[turn {m['turn']}] {m['sender']} -> {m['recipient']}: {m['content']}"
        for m in trace.get("messages", [])
    ]
    actions = [
        f"turn {a['turn']} {a['agent']} {a['name']} {a['payload']}"
        for a in trace.get("actions", [])
    ]
    return (
        f"Task brief.\n{brief}\n\n"
        f"Transcript.\n" + ("\n".join(lines) or "(empty)") + "\n\n"
        f"Operations recorded.\n" + ("\n".join(actions) or "(none)")
    )


def default_testbed_root() -> str | None:
    for candidate in (
        os.path.join("shared_testbed", "data", "raw_runs"),
        os.path.join("..", "minimal-mas-failure-modes", "data", "raw_runs"),
    ):
        if os.path.isdir(candidate) and glob.glob(os.path.join(candidate, "*.jsonl")):
            return candidate
    return None


def iter_items(raw_runs_dir: str, scenario_ids: list[str] | None = None,
               require_decidable: bool = True) -> Iterator[CascadeItem]:
    pattern = os.path.join(raw_runs_dir, "*.jsonl")
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                trace = record.get("trace", {})
                detection = record.get("detection", {})
                scenario_id = trace.get("scenario_id", "")
                if scenario_ids and scenario_id not in scenario_ids:
                    continue
                if require_decidable and not detection.get("decidable", False):
                    continue
                if trace.get("error"):
                    continue
                yield CascadeItem(
                    item_id=f"{os.path.basename(path)[:-6]}#{record.get('replicate')}",
                    ground_truth_is_error=bool(detection.get("occurred", False)),
                    payload={"prompt": render_prompt(trace)},
                    scenario_id=scenario_id,
                    failure_mode_id=FAILURE_MODE_BY_SCENARIO.get(scenario_id, scenario_id),
                    source_path=path,
                    source_config=trace.get("config", {}),
                )


def load_items(raw_runs_dir: str, scenario_ids: list[str] | None = None,
               n_errors: int | None = None, n_non_errors: int | None = None,
               shuffle_seed: int = 20260701) -> list[CascadeItem]:
    import random

    items = list(iter_items(raw_runs_dir, scenario_ids))
    errors = [i for i in items if i.ground_truth_is_error]
    non_errors = [i for i in items if not i.ground_truth_is_error]

    rng = random.Random(shuffle_seed)
    rng.shuffle(errors)
    rng.shuffle(non_errors)

    if n_errors is not None:
        errors = errors[:n_errors]
    if n_non_errors is not None:
        non_errors = non_errors[:n_non_errors]

    selected = errors + non_errors
    rng.shuffle(selected)
    return selected


def summarise(items: list[CascadeItem]) -> dict[str, Any]:
    by_mode: dict[str, dict[str, int]] = {}
    for item in items:
        row = by_mode.setdefault(item.failure_mode_id, {"errors": 0, "non_errors": 0})
        row["errors" if item.ground_truth_is_error else "non_errors"] += 1
    return {
        "n_items": len(items),
        "n_errors": sum(1 for i in items if i.ground_truth_is_error),
        "n_non_errors": sum(1 for i in items if not i.ground_truth_is_error),
        "by_failure_mode": by_mode,
    }
