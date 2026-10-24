import pytest

from app.chunker import chunk_pages, split_sentences
from app.pdf_parser import PageText


def _pages(*texts: str) -> list[PageText]:
    return [PageText(number=i, text=t) for i, t in enumerate(texts, start=1)]


def test_chunks_never_cross_pages():
    sentence = "This sentence fills space on the page. "
    chunks = chunk_pages(_pages(sentence * 30, sentence * 30), max_chars=300, overlap_chars=60)
    for chunk in chunks:
        assert chunk.chunk_id.startswith(f"p{chunk.page}-")
    assert {c.page for c in chunks} == {1, 2}


def test_chunks_respect_max_length_and_overlap():
    text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(60))
    chunks = chunk_pages(_pages(text), max_chars=200, overlap_chars=60)
    assert len(chunks) > 3
    assert all(len(c.text) <= 200 for c in chunks)
    # Consecutive chunks share at least one sentence.
    for a, b in zip(chunks, chunks[1:]):
        last_sentence = split_sentences(a.text)[-1]
        assert last_sentence in b.text


def test_overlong_sentence_is_wrapped():
    chunks = chunk_pages(_pages("word " * 400), max_chars=250, overlap_chars=0)
    assert len(chunks) > 1
    assert all(len(c.text) <= 250 for c in chunks)


def test_chunk_ids_are_unique_and_empty_pages_skipped():
    chunks = chunk_pages(_pages("Alpha text here.", "", "Gamma text here."), max_chars=200, overlap_chars=20)
    assert [c.chunk_id for c in chunks] == ["p1-c0", "p3-c0"]


def test_overlap_must_be_smaller_than_window():
    with pytest.raises(ValueError):
        chunk_pages(_pages("text"), max_chars=100, overlap_chars=100)


def test_split_sentences_keeps_paragraphs_separate():
    assert split_sentences("First one. Second one.\n\nThird para") == ["First one.", "Second one.", "Third para"]
