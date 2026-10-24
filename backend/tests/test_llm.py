"""HTTP-level behaviour of the LLM clients, with the network mocked."""

import httpx
import pytest

from app.config import Settings
from app.errors import LLMError
from app.llm import GeminiClient, OllamaClient, build_llm_client


class _FakePost:
    def __init__(self, response: httpx.Response | Exception):
        self.response = response
        self.calls: list[dict] = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _response(status: int, body: dict) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("POST", "https://example.test"))


def test_gemini_sends_key_in_header_and_skips_thought_parts(monkeypatch):
    fake = _FakePost(_response(200, {"candidates": [{"content": {"parts": [
        {"text": "thinking...", "thought": True},
        {"text": '{"answerable": true}'},
    ]}}]}))
    monkeypatch.setattr(httpx, "post", fake)
    text = GeminiClient("secret-key", "gemini-2.5-flash", 5).generate_json("sys", "prompt")
    assert text == '{"answerable": true}'
    call = fake.calls[0]
    assert call["headers"]["x-goog-api-key"] == "secret-key"
    assert "secret-key" not in call["url"]
    assert call["json"]["generationConfig"]["responseMimeType"] == "application/json"


@pytest.mark.parametrize(
    ("status", "body", "code"),
    [
        (429, {"error": {"message": "quota"}}, "llm_rate_limited"),
        (429, {"error": {"details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}, "llm_quota_exhausted"),
        (400, {"error": {"message": "API key not valid. Please pass a valid API key."}}, "llm_auth"),
        (403, {"error": {"message": "forbidden"}}, "llm_auth"),
        (503, {"error": {"message": "overloaded"}}, "llm_overloaded"),
        (404, {"error": {"message": "model not found"}}, "llm_unavailable"),
        (200, {"promptFeedback": {"blockReason": "SAFETY"}}, "llm_blocked"),
        (200, {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}, "llm_empty"),
    ],
)
def test_gemini_errors_are_mapped(monkeypatch, status, body, code):
    monkeypatch.setattr("app.llm.time.sleep", lambda s: None)
    monkeypatch.setattr(httpx, "post", _FakePost(_response(status, body)))
    with pytest.raises(LLMError) as err:
        GeminiClient("k", "m", 5).generate_json("s", "p")
    assert err.value.code == code


def test_gemini_retries_transient_overload_then_succeeds(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("app.llm.time.sleep", sleeps.append)
    replies = iter([
        _response(503, {"error": {"message": "overloaded"}}),
        _response(200, {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}),
    ])
    monkeypatch.setattr(httpx, "post", lambda url, **kw: next(replies))
    assert GeminiClient("k", "m", 5).generate_json("s", "p") == '{"ok": true}'
    assert sleeps == [2.0]


def test_gemini_does_not_retry_client_errors(monkeypatch):
    fake = _FakePost(_response(429, {"error": {"message": "quota"}}))
    monkeypatch.setattr(httpx, "post", fake)
    with pytest.raises(LLMError):
        GeminiClient("k", "m", 5).generate_json("s", "p")
    assert len(fake.calls) == 1


def test_gemini_timeout(monkeypatch):
    monkeypatch.setattr(httpx, "post", _FakePost(httpx.ReadTimeout("slow")))
    with pytest.raises(LLMError) as err:
        GeminiClient("k", "m", 5).generate_json("s", "p")
    assert err.value.code == "llm_timeout"


def test_ollama_returns_message_content(monkeypatch):
    fake = _FakePost(_response(200, {"message": {"content": '{"answerable": false}'}}))
    monkeypatch.setattr(httpx, "post", fake)
    assert OllamaClient("http://localhost:11434", "llama3.2:3b", 5).generate_json("s", "p") == '{"answerable": false}'
    assert fake.calls[0]["json"]["format"] == "json"


def test_provider_selection(monkeypatch):
    monkeypatch.setattr(OllamaClient, "is_available", lambda self: False)
    assert build_llm_client(Settings(_env_file=None, llm_provider="none", gemini_api_key="k")) is None
    assert build_llm_client(Settings(_env_file=None, llm_provider="auto", gemini_api_key="k")).name == "gemini"
    assert build_llm_client(Settings(_env_file=None, llm_provider="auto")) is None
    assert build_llm_client(Settings(_env_file=None, llm_provider="gemini")) is None
    assert build_llm_client(Settings(_env_file=None, llm_provider="ollama")).name == "ollama"

    monkeypatch.setattr(OllamaClient, "is_available", lambda self: True)
    assert build_llm_client(Settings(_env_file=None, llm_provider="auto")).name == "ollama"
