from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

from .verifier_chain import VerifierChain, CascadeResult


class ScenarioSource(Iterable):
    def __iter__(self):
        raise NotImplementedError


class GateRunner:
    def __init__(
        self,
        scenario_source: Any,
        verifier_chain: VerifierChain,
        output_dir: str,
        pairing_condition: str,
        failure_mode_id: str,
        seed_base: int = 20260701,
    ):
        self.scenario_source = scenario_source
        self.verifier_chain = verifier_chain
        self.output_dir = Path(output_dir)
        self.pairing_condition = pairing_condition
        self.failure_mode_id = failure_mode_id
        self.seed_base = seed_base
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _output_path(self) -> Path:
        fname = (f"{self.failure_mode_id}__{self.pairing_condition}"
                 f"__{self.verifier_chain.verifier_id}.jsonl")
        return self.output_dir / fname

    def run_all(self, k: Optional[int] = None, n_repeats: int = 1,
                progress: bool = True) -> list:
        results = []
        out_path = self._output_path()
        total_items = len(self.scenario_source) if hasattr(self.scenario_source, "__len__") else None
        started = time.time()
        done_items = 0
        with open(out_path, "a", encoding="utf-8") as f:
            for scenario in self.scenario_source:
                item_id = (scenario.get("item_id") if isinstance(scenario, dict)
                           else getattr(scenario, "item_id"))
                ground_truth_is_error = (
                    scenario.get("ground_truth_is_error")
                    if isinstance(scenario, dict)
                    else getattr(scenario, "ground_truth_is_error")
                )
                payload = (scenario.get("payload") if isinstance(scenario, dict)
                           else getattr(scenario, "payload"))
                for rep in range(n_repeats):
                    digest = hashlib.sha256(f"{item_id}|{rep}".encode("utf-8")).digest()
                    seed = self.seed_base + int.from_bytes(digest[:4], "big") % 1_000_003
                    cascade_result: CascadeResult = self.verifier_chain.run(
                        item=payload,
                        item_id=item_id,
                        ground_truth_is_error=ground_truth_is_error,
                        seed_base=seed,
                        k=k,
                    )
                    record = self._to_record(cascade_result, rep)
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    results.append(record)
                done_items += 1
                if progress:
                    elapsed = time.time() - started
                    gates = len(results) * (k or self.verifier_chain.k_max)
                    rate = gates / elapsed if elapsed > 0 else 0.0
                    eta = ""
                    if total_items:
                        remaining = (total_items - done_items) * (k or self.verifier_chain.k_max) * n_repeats
                        eta = f" eta {remaining / rate / 60:.1f}m" if rate > 0 else ""
                        head = f"[{done_items}/{total_items}]"
                    else:
                        head = f"[{done_items}]"
                    print(f"  {head} {self.failure_mode_id} "
                          f"{gates} gates in {elapsed / 60:.1f}m "
                          f"({rate:.2f} gates/s){eta}", file=sys.stderr, flush=True)
        return results

    def _to_record(self, cascade_result: CascadeResult, repeat_index: int) -> dict:
        return {
            "item_id": cascade_result.item_id,
            "repeat_index": repeat_index,
            "pairing_condition": self.pairing_condition,
            "failure_mode_id": self.failure_mode_id,
            "verifier_id": self.verifier_chain.verifier_id,
            "verifier_pool": self.verifier_chain.pool_ids(),
            "ground_truth_is_error": cascade_result.ground_truth_is_error,
            "n_gates": cascade_result.n_gates(),
            "n_decided": len(cascade_result.decided_gates()),
            "n_indeterminate": cascade_result.n_indeterminate(),
            "false_accepts": cascade_result.false_accepts(),
            "survived_all_gates": cascade_result.survived_all_gates(),
            "caught_at": cascade_result.caught_at(),
            "gates": [asdict(g) for g in cascade_result.gates],
        }
