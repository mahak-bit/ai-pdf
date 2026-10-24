"""Shared fixtures.

Unit and API tests use a deterministic hashing embedder and a scripted LLM so they are fast
and offline. These are test doubles only; the application never uses them. Real-model
retrieval is covered separately in test_retrieval_quality.py (marker: ``model``).
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Bag-of-words vectors via feature hashing: shared words -> higher cosine similarity."""

    dimension = 256
    loaded = True

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN.findall(text.lower()):
                if len(token) < 3:
                    continue
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimension
                out[row, bucket] += 1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


class ScriptedLLM:
    """Returns pre-set JSON replies and records the prompts it receives."""

    name = "scripted"
    model = "test"

    def __init__(self, reply: dict | str | Exception | None = None):
        self.reply = reply
        self.prompts: list[str] = []

    def generate_json(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        if isinstance(self.reply, str):
            return self.reply
        if self.reply is not None:
            return json.dumps(self.reply)
        # Default: cite the first source shown in the prompt.
        first = re.search(r"\[(S\d+)\]", prompt)
        sid = first.group(1) if first else "S1"
        return json.dumps({"answerable": True, "answer": f"Per the document [{sid}].", "citations": [sid]})


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(72, 72, 540, 760), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def make_scanned_pdf(page_count: int = 2) -> bytes:
    """Pages that contain only an image, like a scan with no OCR text layer."""
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 80), False)
    pixmap.set_rect(pixmap.irect, (200, 200, 200))
    png = pixmap.tobytes("png")
    doc = pymupdf.open()
    for _ in range(page_count):
        page = doc.new_page()
        page.insert_image(pymupdf.Rect(72, 72, 540, 720), stream=png)
    data = doc.tobytes()
    doc.close()
    return data


def make_encrypted_pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Secret content that needs a password.")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    doc.close()
    return data


HANDBOOK_PAGES = [
    "Welcome to the Orchard Valley Cycling Club. The club was founded in 1987 by four riders.",
    "Membership costs 40 dollars per year. Students pay a reduced fee of 20 dollars per year.",
    "Group rides leave from the town fountain every Saturday at 8 a.m. Helmets are mandatory on every ride.",
]


@pytest.fixture
def handbook_pdf() -> bytes:
    return make_pdf(HANDBOOK_PAGES)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        llm_provider="none",
        max_upload_mb=1,
        max_pages=20,
        min_relevance=0.2,
        top_k=3,
    )


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def make_client(settings, embedder):
    from fastapi.testclient import TestClient

    clients = []

    def _make(llm=None, **overrides):
        s = settings.model_copy(update=overrides) if overrides else settings
        client = TestClient(create_app(s, embedder=embedder, llm=llm, warm_embedder=False))
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)
