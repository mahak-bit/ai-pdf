"""Runtime configuration, read from environment variables (and an optional .env file)."""

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
    gemini_model: str = "gemini-3.1-flash-lite"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    llm_timeout_seconds: float = 60.0

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    data_dir: Path = BACKEND_ROOT / "data"
    max_upload_mb: int = 20
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

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
