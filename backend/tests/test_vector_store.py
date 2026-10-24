import numpy as np
import pytest

from app.chunker import Chunk
from app.vector_store import DocumentIndex

CHUNKS = [
    Chunk("p1-c0", 1, "Membership fees are paid every January."),
    Chunk("p2-c0", 2, "Helmets are mandatory on all group rides."),
    Chunk("p5-c0", 5, "The club was founded in 1987 by four riders."),
]


def test_search_hits_resolve_to_their_source_passage(embedder):
    index = DocumentIndex.build(CHUNKS, embedder.embed([c.text for c in CHUNKS]))
    hits = index.search(embedder.embed(["When was the club founded?"])[0], k=3)
    assert hits[0].chunk.chunk_id == "p5-c0"
    assert hits[0].chunk.page == 5
    assert hits[0].score > hits[-1].score


def test_k_larger_than_index_returns_all_without_padding(embedder):
    index = DocumentIndex.build(CHUNKS, embedder.embed([c.text for c in CHUNKS]))
    hits = index.search(embedder.embed(["helmets"])[0], k=50)
    assert len(hits) == 3
    assert len({h.chunk.chunk_id for h in hits}) == 3


def test_save_and_load_round_trip(tmp_path, embedder):
    vectors = embedder.embed([c.text for c in CHUNKS])
    index = DocumentIndex.build(CHUNKS, vectors)
    index.save(tmp_path / "index.faiss")
    loaded = DocumentIndex.load(tmp_path / "index.faiss", CHUNKS)
    query = embedder.embed(["helmets on rides"])[0]
    assert [h.chunk.chunk_id for h in loaded.search(query, 3)] == [h.chunk.chunk_id for h in index.search(query, 3)]


def test_load_rejects_mismatched_chunk_list(tmp_path, embedder):
    DocumentIndex.build(CHUNKS, embedder.embed([c.text for c in CHUNKS])).save(tmp_path / "index.faiss")
    with pytest.raises(ValueError):
        DocumentIndex.load(tmp_path / "index.faiss", CHUNKS[:2])


def test_build_requires_one_vector_per_chunk():
    with pytest.raises(ValueError):
        DocumentIndex.build(CHUNKS, np.zeros((2, 8), dtype=np.float32))
