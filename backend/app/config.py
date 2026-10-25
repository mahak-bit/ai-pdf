"""Runtime configuration, read from environment variables (and an optional .env file)."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM provider. "auto" picks Gemini when a key is present, otherwise Ollama when
    # reachable, otherwise retrieval-only mode (passages are shown, no answer is generated).
    llm_provider: Literal["auto", "gemini", "ollama", "none"] = "auto"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    # Tried in order when the main model is overloaded, rate-limited or out of daily quota.
    gemini_fallback_models: str = "gemini-3.1-flash-lite,gemini-flash-lite-latest"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    llm_timeout_seconds: float = 30.0

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    data_dir: Path = BACKEND_ROOT / "data"
    max_upload_mb: int = 20  # through the API (capped at 4 MB on Vercel)
    max_direct_upload_mb: int = 20  # browser-to-Blob uploads, when Blob storage is enabled
    max_pages: int = 500

    # Chunking: page-scoped, sentence-packed windows. ~600 chars is ~130 tokens, close to
    # the passage length MiniLM was trained on (it truncates input at 256 word pieces).
    chunk_max_chars: int = 600
    chunk_overlap_chars: int = 120

    # Retrieval.
    top_k: int = 5
    # Cosine similarity below which a passage is not treated as evidence. This is only a
    # coarse off-topic filter: on the sample eval set, off-topic questions scored <= 0.08
    # and the weakest answerable one 0.17, while in-domain unanswerable questions overlap
    # answerable ones (0.26-0.46), so those are left to the LLM's grounding check.
    # Re-tune for your documents with scripts/evaluate.py.
    min_relevance: float = 0.12

    cors_origins: str = "http://localhost:3000"

    # Vercel Blob mirror for documents (see blob_mirror.py). Connecting a store to a project
    # sets BLOB_STORE_ID (the SDK then authenticates with Vercel OIDC) or, for older stores,
    # BLOB_READ_WRITE_TOKEN. Either one enables the mirror.
    blob_read_write_token: str | None = None
    blob_store_id: str | None = None
    blob_access: Literal["private", "public"] = "private"

    @property
    def blob_enabled(self) -> bool:
        return bool(self.blob_read_write_token or self.blob_store_id)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


def _platform_defaults(settings: Settings) -> Settings:
    """On Vercel only /tmp is writable and request bodies are capped at 4.5 MB."""
    if not os.environ.get("VERCEL"):
        return settings
    # Hugging Face libraries write caches under HOME, which is read-only on Vercel.
    os.environ.setdefault("HF_HOME", "/tmp/huggingface")
    updates: dict = {}
    if "DATA_DIR" not in os.environ:
        updates["data_dir"] = Path("/tmp/documind")
    if "MAX_UPLOAD_MB" not in os.environ:
        updates["max_upload_mb"] = 4
    return settings.model_copy(update=updates) if updates else settings


@lru_cache
def get_settings() -> Settings:
    return _platform_defaults(Settings())
