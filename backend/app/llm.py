"""LLM providers.

Each client offers ``generate_json`` (one JSON object, used by the non-streaming endpoint
and the evaluation script) and ``stream_text`` (plain text as it is generated, used by
the streaming endpoint). Gemini is called through its REST API with the key sent in a
header from the server only. Ollama is the local, free fallback. When neither is
available the app runs in retrieval-only mode and never invents an answer.
"""

import json
import logging
import time
from collections.abc import Callable, Iterator
from typing import Protocol

import httpx

from .config import Settings
from .errors import LLMError

log = logging.getLogger(__name__)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_STREAM_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
_RETRYABLE_STATUS = {500, 502, 503, 504}
_RETRY_BACKOFF_SECONDS = (2.0, 5.0)


class LLMClient(Protocol):
    name: str
    model: str

    # ``on_model`` is called with the model that actually produced the reply, which can
    # differ from ``model`` when a fallback model was used.
    def generate_json(self, system: str, prompt: str, on_model: Callable[[str], None] | None = None) -> str: ...

    def stream_text(self, system: str, prompt: str, on_model: Callable[[str], None] | None = None) -> Iterator[str]: ...


# Errors where trying a different model can help: capacity, quota and slowness are per model.
_FALLBACK_CODES = {"llm_overloaded", "llm_quota_exhausted", "llm_rate_limited", "llm_timeout"}


class GeminiClient:
    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout: float, fallback_models: tuple[str, ...] = ()):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.models = [model, *dict.fromkeys(m for m in fallback_models if m and m != model)]

    def _body(self, system: str, prompt: str, json_output: bool) -> dict:
        config: dict = {"temperature": 0.1}
        if json_output:
            config["responseMimeType"] = "application/json"
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": config,
        }

    def generate_json(self, system: str, prompt: str, on_model: Callable[[str], None] | None = None) -> str:
        body = self._body(system, prompt, json_output=True)
        for i, model in enumerate(self.models):
            try:
                text = self._generate_one(model, body, last=i + 1 == len(self.models))
            except LLMError as exc:
                if exc.code in _FALLBACK_CODES and i + 1 < len(self.models):
                    log.warning("Gemini %s failed (%s); falling back to %s", model, exc.code, self.models[i + 1])
                    continue
                raise
            if on_model:
                on_model(model)
            return text
        raise AssertionError("unreachable")

    def stream_text(self, system: str, prompt: str, on_model: Callable[[str], None] | None = None) -> Iterator[str]:
        """Yield answer text as Gemini generates it. Falls back to the next model only if
        the current one fails before producing any text."""
        body = self._body(system, prompt, json_output=False)
        for i, model in enumerate(self.models):
            produced = False
            try:
                for text in self._stream_one(model, body, last=i + 1 == len(self.models)):
                    if not produced:
                        produced = True
                        if on_model:
                            on_model(model)
                    yield text
                return
            except LLMError as exc:
                if not produced and exc.code in _FALLBACK_CODES and i + 1 < len(self.models):
                    log.warning("Gemini %s failed (%s); falling back to %s", model, exc.code, self.models[i + 1])
                    continue
                raise

    def _generate_one(self, model: str, body: dict, last: bool) -> str:
        for attempt, backoff in enumerate((*_backoffs(last), None)):
            try:
                response = httpx.post(
                    GEMINI_URL.format(model=model), headers={"x-goog-api-key": self.api_key}, json=body, timeout=self.timeout
                )
            except httpx.TimeoutException as exc:
                raise LLMError("Gemini took too long to respond. Try again.", code="llm_timeout") from exc
            except httpx.HTTPError as exc:
                raise LLMError("Could not reach the Gemini API. Check the server's network connection.") from exc
            # 5xx from Gemini is usually transient overload ("model is overloaded"): retry.
            if response.status_code not in _RETRYABLE_STATUS or backoff is None:
                break
            log.info("Gemini %s returned %s; retry %d in %.1fs", model, response.status_code, attempt + 1, backoff)
            time.sleep(backoff)
        self._raise_for_status(response, model)
        return _candidate_text(response.json(), required=True)

    def _stream_one(self, model: str, body: dict, last: bool) -> Iterator[str]:
        url = GEMINI_STREAM_URL.format(model=model)
        for attempt, backoff in enumerate((*_backoffs(last), None)):
            try:
                with httpx.stream("POST", url, headers={"x-goog-api-key": self.api_key}, json=body, timeout=self.timeout) as response:
                    if response.status_code in _RETRYABLE_STATUS and backoff is not None:
                        log.info("Gemini %s returned %s; retry %d in %.1fs", model, response.status_code, attempt + 1, backoff)
                        time.sleep(backoff)
                        continue
                    if response.status_code >= 400:
                        response.read()
                        self._raise_for_status(response, model)
                    produced = False
                    for line in response.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        text = _candidate_text(json.loads(line[5:]), required=False)
                        if text:
                            produced = True
                            yield text
                    if not produced:
                        raise LLMError("Gemini returned an empty answer.", code="llm_empty")
                    return
            except httpx.TimeoutException as exc:
                raise LLMError("Gemini took too long to respond. Try again.", code="llm_timeout") from exc
            except httpx.HTTPError as exc:
                raise LLMError("Could not reach the Gemini API. Check the server's network connection.") from exc

    def _raise_for_status(self, response: httpx.Response, model: str) -> None:
        if response.status_code < 400:
            return
        if response.status_code in _RETRYABLE_STATUS:
            raise LLMError("Gemini is overloaded right now. Try again in a minute.", code="llm_overloaded", status_code=503)
        if response.status_code == 429:
            if "PerDay" in response.text:
                raise LLMError(
                    f"The daily Gemini quota for {model} is used up. It resets within 24 hours; "
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
        log.warning("Gemini error %s: %s", response.status_code, response.text[:500])
        raise LLMError(f"Gemini returned an error (HTTP {response.status_code}). Try again shortly.")


def _backoffs(last_model: bool) -> tuple[float, ...]:
    """Retry an overloaded model twice when it is the last option, once when another can take over."""
    return _RETRY_BACKOFF_SECONDS if last_model else _RETRY_BACKOFF_SECONDS[:1]


def _candidate_text(payload: dict, required: bool) -> str:
    """Extract answer text from a Gemini response (or one streamed chunk), skipping thoughts."""
    candidates = payload.get("candidates") or []
    if not candidates:
        reason = (payload.get("promptFeedback") or {}).get("blockReason")
        if reason or required:
            raise LLMError(f"Gemini returned no answer (blocked: {reason or 'unknown'}).", code="llm_blocked")
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if required and not text.strip():
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

    def generate_json(self, system: str, prompt: str, on_model: Callable[[str], None] | None = None) -> str:
        if on_model:
            on_model(self.model)
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

    def stream_text(self, system: str, prompt: str, on_model: Callable[[str], None] | None = None) -> Iterator[str]:
        if on_model:
            on_model(self.model)
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": True,
            "options": {"temperature": 0.1},
        }
        try:
            with httpx.stream("POST", f"{self.base_url}/api/chat", json=body, timeout=self.timeout) as response:
                if response.status_code >= 400:
                    response.read()
                    raise LLMError(f"Ollama returned an error: {_error_message(response)}")
                for line in response.iter_lines():  # newline-delimited JSON
                    if line.strip():
                        text = (json.loads(line).get("message") or {}).get("content", "")
                        if text:
                            yield text
        except httpx.TimeoutException as exc:
            raise LLMError("The local Ollama model took too long to respond.", code="llm_timeout") from exc
        except httpx.HTTPError as exc:
            raise LLMError("Could not reach Ollama. Is `ollama serve` running?") from exc


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
        return GeminiClient(
            settings.gemini_api_key,
            settings.gemini_model,
            settings.llm_timeout_seconds,
            fallback_models=tuple(m.strip() for m in settings.gemini_fallback_models.split(",")),
        )
    if provider == "gemini":
        log.warning("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set; running retrieval-only")
        return None
    ollama = OllamaClient(settings.ollama_base_url, settings.ollama_model, settings.llm_timeout_seconds)
    if provider == "ollama" or ollama.is_available():
        return ollama
    log.info("No LLM configured (no GEMINI_API_KEY, Ollama not reachable); running retrieval-only")
    return None
