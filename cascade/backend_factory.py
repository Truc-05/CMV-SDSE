"""One place that turns a VerifierSpec into a live backend, dispatched by provider.

Adding a new verifier provider later is a single branch here plus a config block under
`backend:` in models.yaml -- the run loop and the rest of the pipeline stay untouched.

Config shape (models.yaml):

    backend:
      timeout_s: 120
      max_retries: 3
      retry_backoff_s: 2.0
      ollama:
        host: "http://localhost:11437"

Legacy flat `host:` (no `ollama:` block) is still honoured.
"""
from __future__ import annotations

from typing import Any

from .ollama_backend import BackendError, OllamaVerifierBackend, VerifierSpec


def build_backend(spec: VerifierSpec, settings: dict[str, Any], thresholds: dict[str, Any],
                  system_prompt: str):
    """Return a ready backend for `spec`, choosing the class from `spec.provider`."""
    provider = getattr(spec, "provider", "ollama")
    timeout_s = float(settings.get("timeout_s", 120))
    max_retries = int(thresholds.get("retry_on_backend_error", 3))
    retry_backoff_s = float(settings.get("retry_backoff_s", 2.0))

    if provider == "ollama":
        ollama = settings.get("ollama", {}) if isinstance(settings.get("ollama"), dict) else {}
        host = ollama.get("host", settings.get("host", "http://localhost:11434"))
        return OllamaVerifierBackend(
            spec=spec, system_prompt=system_prompt, host=host,
            timeout_s=timeout_s, max_retries=max_retries, retry_backoff_s=retry_backoff_s,
        )

    raise BackendError(
        f"verifier {spec.id!r} has unknown provider {provider!r}; expected 'ollama'."
    )
