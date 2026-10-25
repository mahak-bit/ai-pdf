"""Retrieval with the real all-MiniLM-L6-v2 model on the bundled sample PDF.

Downloads the model (~90 MB) on first run. Skip with:  pytest -m "not model"
"""

import pytest

from app.chunker import chunk_pages
from app.config import Settings
from app.embeddings import SentenceTransformerEmbedder
from app.main import SAMPLE_PDF
from app.pdf_parser import extract_pages
from app.rag import RAGPipeline, SearchDoc
from app.vector_store import DocumentIndex

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def pipeline_and_index():
    settings = Settings(_env_file=None)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    parsed = extract_pages(SAMPLE_PDF.read_bytes(), max_pages=50)
    chunks = chunk_pages(parsed.pages, settings.chunk_max_chars, settings.chunk_overlap_chars)
    index = DocumentIndex.build(chunks, embedder.embed([c.text for c in chunks]))
    return RAGPipeline(embedder, None, settings.top_k, settings.min_relevance), [SearchDoc("sample000000", SAMPLE_PDF.name, index)]


@pytest.mark.parametrize(
    ("question", "page"),
    [
        ("How long can I borrow a DVD?", 3),
        ("Are there late fees for overdue books?", 4),
        ("What is the Wi-Fi network called?", 5),
        ("What time does Central Library open on Sunday?", 6),
        ("How much does a non-resident card cost?", 2),
    ],
)
def test_expected_page_is_top_ranked(pipeline_and_index, question, page):
    pipeline, index = pipeline_and_index
    hits = pipeline.retrieve(index, question)
    assert hits, "expected at least one passage above the relevance floor"
    assert hits[0].page == page


@pytest.mark.parametrize(
    "question",
    ["What is the boiling point of mercury?", "Who won the 2018 football World Cup?"],
)
def test_off_topic_questions_fall_below_relevance_floor(pipeline_and_index, question):
    pipeline, index = pipeline_and_index
    assert pipeline.retrieve(index, question) == []
