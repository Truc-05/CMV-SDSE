from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


class VerifierBackend(Protocol):
    def verify(self, item: Any, seed: int) -> bool:
        ...


@dataclass
class GateResult:
    gate_index: int
    accepted: Optional[bool]
    latency_seconds: float
    seed: int
    verifier_id: str
    raw_response: Optional[str] = None
    error: Optional[str] = None


@dataclass
class CascadeResult:
    item_id: str
    ground_truth_is_error: bool
    gates: list = field(default_factory=list)

    def n_gates(self) -> int:
        return len(self.gates)

    def decided_gates(self) -> list:
        return [g for g in self.gates if g.accepted is not None]

    def n_indeterminate(self) -> int:
        return sum(1 for g in self.gates if g.accepted is None)

    def false_accepts(self) -> int:
        if not self.ground_truth_is_error:
            return 0
        return sum(1 for g in self.decided_gates() if g.accepted)

    def survived_all_gates(self) -> bool:
        if not self.ground_truth_is_error:
            return False
        decided = self.decided_gates()
        return bool(decided) and all(g.accepted for g in decided)

    def caught_at(self) -> Optional[int]:
        if not self.ground_truth_is_error:
            return None
        for g in self.gates:
            if g.accepted is False:
                return g.gate_index
        return None


class VerifierChain:
    def __init__(
        self,
        verifier_backend: Any,
        verifier_id: str,
        k_max: int,
        retry_on_backend_error: int = 3,
        timeout_seconds_per_gate: float = 60.0,
    ):
        self.pool = (list(verifier_backend)
                     if isinstance(verifier_backend, (list, tuple))
                     else [verifier_backend])
        if not self.pool:
            raise ValueError("verifier pool is empty")
        self.verifier_backend = self.pool[0]
        self.verifier_id = verifier_id
        self.k_max = k_max
        self.retry_on_backend_error = retry_on_backend_error
        self.timeout_seconds_per_gate = timeout_seconds_per_gate

    def backend_for_gate(self, gate_index: int) -> Any:
        return self.pool[(gate_index - 1) % len(self.pool)]

    def pool_ids(self) -> list:
        return [getattr(b, "verifier_id", self.verifier_id) for b in self.pool]

    def _run_single_gate(self, item: Any, gate_index: int, seed: int) -> GateResult:
        rng = random.Random(seed)
        backend = self.backend_for_gate(gate_index)
        gate_verifier_id = getattr(backend, "verifier_id", self.verifier_id)
        last_error = None
        start = time.monotonic()
        for attempt in range(self.retry_on_backend_error):
            start = time.monotonic()
            try:
                accepted = backend.verify(item, seed=rng.randint(0, 2**31 - 1))
                elapsed = time.monotonic() - start
                return GateResult(
                    gate_index=gate_index,
                    accepted=accepted,
                    latency_seconds=elapsed,
                    seed=seed,
                    verifier_id=gate_verifier_id,
                )
            except Exception as exc:
                last_error = str(exc)
                continue
        elapsed = time.monotonic() - start
        return GateResult(
            gate_index=gate_index,
            accepted=None,
            latency_seconds=elapsed,
            seed=seed,
            verifier_id=gate_verifier_id,
            error=last_error,
        )

    def run(
        self,
        item: Any,
        item_id: str,
        ground_truth_is_error: bool,
        seed_base: int,
        k: Optional[int] = None,
        stop_early_on_first_reject: bool = False,
    ) -> CascadeResult:
        k = k or self.k_max
        result = CascadeResult(item_id=item_id, ground_truth_is_error=ground_truth_is_error)
        for gate_index in range(1, k + 1):
            gate_seed = seed_base + gate_index * 7919
            gate_result = self._run_single_gate(item, gate_index, gate_seed)
            result.gates.append(gate_result)
            if stop_early_on_first_reject and gate_result.accepted is False:
                break
        return result
