"""Sentence embeddings. Vectors are L2-normalised so inner product equals cosine similarity."""

import threading
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dimension: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dimension) with unit-norm rows."""
        ...


class SentenceTransformerEmbedder:
    """all-MiniLM-L6-v2 by default: 384 dimensions, ~90 MB, runs acceptably on CPU."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                # Prefer the local cache: otherwise every start makes network calls to the
                # Hugging Face Hub, which can stall uploads for minutes when the Hub is slow.
                try:
                    self._model = SentenceTransformer(self.model_name, device="cpu", local_files_only=True)
                except Exception:
                    self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    @property
    def dimension(self) -> int:
        model = self._load()
        get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return int(get_dim())

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)
