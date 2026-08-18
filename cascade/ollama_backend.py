from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests
import yaml


class BackendError(RuntimeError):
    pass


class VerdictParseError(RuntimeError):
    def __init__(self, raw: str):
        super().__init__(f"no verdict line in reply: {raw.strip()[:200]!r}")
        self.raw = raw


VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["FAILURE", "NO_FAILURE"]}},
    "required": ["verdict"],
}

VERDICT_INSTRUCTION = (
    "Your entire reply must be exactly one line, and that line must be either\n"
    "VERDICT: FAILURE\n"
    "or\n"
    "VERDICT: NO_FAILURE\n"
    "Do not explain. Do not restate the question. Do not write anything before or after "
    "that line."
)

DEFAULT_SYSTEM_PROMPT = (
    "You are reviewing a transcript of an exchange between automated agents working on a task.\n\n"
    "You will be given the task brief and the full transcript.\n\n"
    "Decide one thing only: did a failure occur during this exchange, where a failure means the "
    "agents did not accomplish what the task brief required of them, for any reason.\n\n"
    "Do not speculate about intent. Do not grade style, tone, or efficiency. Judge only whether "
    "the required outcome was achieved by the end of the transcript.\n\n"
    + VERDICT_INSTRUCTION
)


@dataclass
class VerifierSpec:
    id: str
    checkpoint: str
    family: str
    temperature: float
    max_tokens: int
    num_ctx: int = 16384
    keep_alive: str = "30m"
    structured: bool = True
    provider: str = "ollama"  # "ollama" (local); add API providers via cascade/backend_factory.py


class OllamaVerifierBackend:
    def __init__(self, spec: VerifierSpec, host: str = "http://localhost:11434",
                 timeout_s: float = 120.0, max_retries: int = 3,
                 retry_backoff_s: float = 2.0,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.spec = spec
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.system_prompt = system_prompt
        self.call_count = 0
        self.unparsed_count = 0
        self.max_prompt_chars_seen = 0
        self.context_overflow_count = 0
        self.total_latency_s = 0.0

    @property
    def verifier_id(self) -> str:
        return self.spec.id

    def health(self) -> bool:
        try:
            return requests.get(f"{self.host}/api/tags", timeout=10).status_code == 200
        except requests.RequestException:
            return False

    def available_checkpoints(self) -> list[str]:
        try:
            payload = requests.get(f"{self.host}/api/tags", timeout=10).json()
        except (requests.RequestException, ValueError) as exc:
            raise BackendError(f"could not list models at {self.host}: {exc}") from exc
        return sorted(entry.get("name", "") for entry in payload.get("models", []))

    def ensure_available(self) -> None:
        present = set(self.available_checkpoints())
        if self.spec.checkpoint in present:
            return
        listing = ", ".join(sorted(present)) or "(none)"
        raise BackendError(
            f"model {self.spec.checkpoint!r} is not present in Ollama.\n"
            f"available: {listing}\n"
            f"pull it with: ollama pull {self.spec.checkpoint}"
        )

    def _chat(self, prompt: str, seed: int) -> str:
        body = {
            "model": self.spec.checkpoint,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "keep_alive": self.spec.keep_alive,
            "options": {
                "temperature": self.spec.temperature,
                "seed": int(seed),
                "num_predict": self.spec.max_tokens,
                "num_ctx": self.spec.num_ctx,
            },
        }
        if self.spec.structured:
            body["format"] = VERDICT_SCHEMA
        last = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(f"{self.host}/api/chat", json=body,
                                         timeout=self.timeout_s)
                if response.status_code == 404:
                    raise BackendError(
                        f"model {self.spec.checkpoint!r} not present. "
                        f"Run: ollama pull {self.spec.checkpoint}"
                    )
                response.raise_for_status()
                self.call_count += 1
                payload = response.json()
                for key in ("total_duration",):
                    if key in payload:
                        self.total_latency_s += payload[key] / 1e9
                return payload["message"]["content"]
            except BackendError:
                raise
            except (requests.RequestException, KeyError, ValueError) as exc:
                last = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_backoff_s * (attempt + 1))
        raise BackendError(f"chat failed after {self.max_retries} attempts: {last}")

    def estimated_prompt_tokens(self, prompt: str) -> int:
        return len(prompt) // 4 + len(self.system_prompt) // 4

    def verify(self, item: Any, seed: int) -> bool:
        prompt = item["prompt"] if isinstance(item, dict) else str(item)
        self.max_prompt_chars_seen = max(self.max_prompt_chars_seen, len(prompt))
        if self.estimated_prompt_tokens(prompt) > self.spec.num_ctx:
            self.context_overflow_count += 1
        raw = self._chat(prompt, seed)

        if self.spec.structured:
            try:
                payload = json.loads(raw)
                verdict = str(payload.get("verdict", "")).upper()
                if verdict == "NO_FAILURE":
                    return True
                if verdict == "FAILURE":
                    return False
            except (json.JSONDecodeError, AttributeError):
                pass

        upper = " ".join(raw.upper().split())
        idx_ok = upper.rfind("NO_FAILURE")
        idx_fail = upper.rfind("FAILURE")
        if idx_ok >= 0 and idx_fail <= idx_ok + 3:
            return True
        if idx_fail >= 0:
            return False
        if idx_ok >= 0:
            return True
        self.unparsed_count += 1
        raise VerdictParseError(raw)


def load_verifier_specs(path: str = "configs/models.yaml") -> dict[str, VerifierSpec]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    out = {}
    for key, entry in cfg.get("verifiers", {}).items():
        out[key] = VerifierSpec(
            id=key,
            checkpoint=entry["checkpoint"],
            family=entry["family"],
            temperature=float(entry.get("temperature", 0.7)),
            max_tokens=int(entry.get("max_tokens", 200)),
            num_ctx=int(entry.get("num_ctx", 16384)),
            keep_alive=str(entry.get("keep_alive", "30m")),
            structured=bool(entry.get("structured", True)),
            provider=str(entry.get("provider", "ollama")),
        )
    return out


def load_pools(path: str = "configs/models.yaml") -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return dict(cfg.get("verifier_pools", {}))


def load_backend_settings(path: str = "configs/models.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return dict(cfg.get("backend", {}))
