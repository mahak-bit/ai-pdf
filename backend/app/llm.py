"""LLM providers. Both return the raw text of a JSON object produced by the model.

Gemini is called through its REST API with the key sent in a header from the server only.
Ollama is the local, free fallback. When neither is available the app runs in
retrieval-only mode and never invents an answer.
"""

import logging
import time
from typing import Protocol

import httpx

from .config import Settings
from .errors import LLMError

log = logging.getLogger(__name__)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_RETRYABLE_STATUS = {500, 502, 503, 504}
_RETRY_BACKOFF_SECONDS = (2.0, 5.0)


class LLMClient(Protocol):
    name: str
    model: str

    def generate_json(self, system: str, prompt: str) -> str: ...


class GeminiClient:
    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout: float):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate_json(self, system: str, prompt: str) -> str:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
        }
        for attempt, backoff in enumerate((*_RETRY_BACKOFF_SECONDS, None)):
            try:
                response = httpx.post(
                    GEMINI_URL.format(model=self.model),
                    headers={"x-goog-api-key": self.api_key},
                    json=body,
                    timeout=self.timeout,
                )
            except httpx.TimeoutException as exc:
                raise LLMError("Gemini took too long to respond. Try again.", code="llm_timeout") from exc
            except httpx.HTTPError as exc:
                raise LLMError("Could not reach the Gemini API. Check the server's network connection.") from exc
            # 5xx from Gemini is usually transient overload ("model is overloaded"): retry.
            if response.status_code not in _RETRYABLE_STATUS or backoff is None:
                break
            log.info("Gemini returned %s; retry %d in %.1fs", response.status_code, attempt + 1, backoff)
            time.sleep(backoff)

        if response.status_code in _RETRYABLE_STATUS:
            raise LLMError("Gemini is overloaded right now. Try again in a minute.", code="llm_overloaded", status_code=503)
        if response.status_code == 429:
            if "PerDay" in response.text:
                raise LLMError(
                    f"The daily Gemini quota for {self.model} is used up. It resets within 24 hours; "
                    "or set GEMINI_MODEL to another model on the server.",
                    code="llm_quota_exhausted",
                    status_code=503,
                )
            raise LLMError("The Gemini free-tier rate limit was reached. Wait a minute and try again.", code="llm_rate_limited", status_code=503)
        if response.status_code in (400, 401, 403):
            detail = _error_message(response)
            log.warning("Gemini rejected the request (%s): %s", response.status_code, detail)
            if "API key" in detail or response.status_code in (401, 403):
                raise LLMError("The Gemini API key was rejected. Check GEMINI_API_KEY on the server.", code="llm_auth")
            raise LLMError(f"Gemini rejected the request: {detail}")
        if response.status_code >= 400:
            log.warning("Gemini error %s: %s", response.status_code, response.text[:500])
            raise LLMError(f"Gemini returned an error (HTTP {response.status_code}). Try again shortly.")

        payload = response.json()
        candidates = payload.get("candidates") or []
        if not candidates:
            reason = (payload.get("promptFeedback") or {}).get("blockReason", "unknown")
            raise LLMError(f"Gemini returned no answer (blocked: {reason}).", code="llm_blocked")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            reason = candidates[0].get("finishReason", "unknown")
            raise LLMError(f"Gemini returned an empty answer (finish reason: {reason}).", code="llm_empty")
        return text


class OllamaClient:
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        names = {m.get("name", "") for m in response.json().get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    def generate_json(self, system: str, prompt: str) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }
        try:
            response = httpx.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout)
        except httpx.TimeoutException as exc:
            raise LLMError("The local Ollama model took too long to respond.", code="llm_timeout") from exc
        except httpx.HTTPError as exc:
            raise LLMError("Could not reach Ollama. Is `ollama serve` running?") from exc
        if response.status_code >= 400:
            raise LLMError(f"Ollama returned an error: {_error_message(response)}")
        return (response.json().get("message") or {}).get("content", "")


def _error_message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:300]
    error = data.get("error")
    if isinstance(error, dict):
        return str(error.get("message", error))
    return str(error or data)[:300]


def build_llm_client(settings: Settings) -> LLMClient | None:
    """Pick a provider. Returns None for retrieval-only mode."""
    provider = settings.llm_provider
    if provider == "none":
        return None
    if provider in ("gemini", "auto") and settings.gemini_api_key:
        return GeminiClient(settings.gemini_api_key, settings.gemini_model, settings.llm_timeout_seconds)
    if provider == "gemini":
        log.warning("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set; running retrieval-only")
        return None
    ollama = OllamaClient(settings.ollama_base_url, settings.ollama_model, settings.llm_timeout_seconds)
    if provider == "ollama" or ollama.is_available():
        return ollama
    log.info("No LLM configured (no GEMINI_API_KEY, Ollama not reachable); running retrieval-only")
    return None
