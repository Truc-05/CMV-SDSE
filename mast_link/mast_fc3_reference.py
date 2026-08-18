from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Optional

DEFAULT_PATH = os.path.join("data", "mast_data_reference", "fc3_reference.json")


@dataclass
class MASTFC3Reference:
    fc3_label: str
    fc3_description: str
    fc3_rate: float
    n_traces_total: Optional[int]
    n_fc3_traces: Optional[int]
    related_failure_modes: list
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_mast_fc3_reference(path: Optional[str] = None) -> MASTFC3Reference:
    target = path or DEFAULT_PATH
    if not os.path.exists(target):
        raise FileNotFoundError(
            f"{target} not found; the FC3 reference describes the MAST verification-failure "
            "share used for comparison"
        )
    with open(target, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return MASTFC3Reference(
        fc3_label=payload.get("fc3_label", "FC3"),
        fc3_description=payload.get("fc3_description", ""),
        fc3_rate=float(payload["fc3_rate"]),
        n_traces_total=payload.get("n_traces_total"),
        n_fc3_traces=payload.get("n_fc3_traces"),
        related_failure_modes=list(payload.get("related_failure_modes", [])),
        source=payload.get("source", "MAST-Data"),
    )
