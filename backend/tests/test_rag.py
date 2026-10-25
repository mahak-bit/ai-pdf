import pytest

from app.chunker import Chunk
from app.errors import LLMError
from app.rag import (
    ABSTAIN_NO_EVIDENCE,
    ABSTAIN_UNGROUNDED,
    RAGPipeline,
    SearchDoc,
    Source,
    build_prompt,
    parse_model_json,
    validate_citations,
)
from app.vector_store import DocumentIndex

from conftest import ScriptedLLM

CHUNKS = [
    Chunk("p1-c0", 1, "The Orchard Valley Cycling Club was founded in 1987 by four riders."),
    Chunk("p2-c0", 2, "Membership costs 40 dollars per year and students pay 20 dollars."),
    Chunk("p3-c0", 3, "Group rides leave from the town fountain every Saturday morning."),
]
OTHER_CHUNKS = [
    Chunk("p1-c0", 1, "The rowing club charges 90 dollars per year for membership."),
    Chunk("p2-c0", 2, "Boats are stored in the boathouse beside the river."),
]


@pytest.fixture
def docs(embedder):
    return [SearchDoc("aaaaaaaaaaaa", "cycling.pdf", DocumentIndex.build(CHUNKS, embedder.embed([c.text for c in CHUNKS])))]


@pytest.fixture
def two_docs(embedder, docs):
    rowing = DocumentIndex.build(OTHER_CHUNKS, embedder.embed([c.text for c in OTHER_CHUNKS]))
    return [*docs, SearchDoc("bbbbbbbbbbbb", "rowing.pdf", rowing)]


def _sources(*pages: int, doc_id: str = "aaaaaaaaaaaa", filename: str = "a.pdf") -> list[Source]:
    return [Source(f"S{i}", f"p{p}-c0", p, f"text {p}", 0.9, doc_id, filename) for i, p in enumerate(pages, start=1)]


# ---- citation validation ---------------------------------------------------------------


def test_validate_citations_keeps_only_provided_ids():
    answer, cited = validate_citations("Fees are 40 dollars [S1]. Also true [S7].", ["S1", "S7", "S9"], _sources(2, 3))
    assert [s.source_id for s in cited] == ["S1"]
    assert "[S7]" not in answer
    assert answer == "Fees are 40 dollars [S1]. Also true."


def test_validate_citations_orders_by_first_appearance_and_dedupes():
    _, cited = validate_citations("A [S2]. B [S1][S2].", ["S1"], _sources(1, 2))
    assert [s.source_id for s in cited] == ["S2", "S1"]


def test_validate_citations_accepts_declared_list_without_markers():
    _, cited = validate_citations("Fees are 40 dollars.", ["[S1]"], _sources(2))
    assert [s.page for s in cited] == [2]


def test_validate_citations_ignores_non_list_declaration():
    _, cited = validate_citations("Text.", "S1", _sources(2))
    assert cited == []


# ---- model output parsing and prompts ---------------------------------------------------


def test_parse_model_json_handles_code_fences():
    assert parse_model_json('```json\n{"answerable": true, "answer": "x"}\n```')["answer"] == "x"


def test_parse_model_json_rejects_garbage():
    with pytest.raises(LLMError):
        parse_model_json("I think the answer is 42.")
    with pytest.raises(LLMError):
        parse_model_json('["not", "an", "object"]')


def test_prompt_labels_sources_with_pages():
    prompt = build_prompt("How much?", _sources(4))
    assert "[S1] (page 4)" in prompt
    assert prompt.endswith("Question: How much?")


def test_prompt_names_documents_when_several_are_searched():
    sources = _sources(4) + [Source("S2", "p1-c0", 1, "x", 0.5, "bbbbbbbbbbbb", "rowing.pdf")]
    prompt = build_prompt("How much?", sources)
    assert "[S1] (a.pdf, page 4)" in prompt
    assert "[S2] (rowing.pdf, page 1)" in prompt


# ---- pipeline: JSON mode ----------------------------------------------------------------


def test_answer_returns_validated_citations_with_pages(embedder, docs):
    llm = ScriptedLLM({"answerable": True, "answer": "It costs 40 dollars a year [S1].", "citations": ["S1"]})
    result = RAGPipeline(embedder, llm, top_k=3, min_relevance=0.1).answer(docs, "How much does membership cost per year?")
    assert result.status == "answered"
    assert result.sources[0].page == 2
    assert result.sources[0].filename == "cycling.pdf"
    assert "40 dollars" in result.sources[0].excerpt
    assert result.model == "scripted:test"


def test_retrieval_merges_documents_by_similarity(embedder, two_docs):
    sources = RAGPipeline(embedder, None, top_k=3, min_relevance=0.05).retrieve(two_docs, "How much is the yearly membership?")
    assert len(sources) <= 3
    assert {s.filename for s in sources} == {"cycling.pdf", "rowing.pdf"}
    assert [s.source_id for s in sources] == [f"S{i}" for i in range(1, len(sources) + 1)]
    assert sources == sorted(sources, key=lambda s: s.score, reverse=True)


def test_abstains_without_calling_llm_when_nothing_is_relevant(embedder, docs):
    llm = ScriptedLLM()
    result = RAGPipeline(embedder, llm, top_k=3, min_relevance=0.2).answer(docs, "Quantum chromodynamics lattice gauge theory?")
    assert result.status == "abstained"
    assert result.answer == ABSTAIN_NO_EVIDENCE
    assert result.sources == []
    assert llm.prompts == []


def test_abstains_when_model_says_unanswerable(embedder, docs):
    llm = ScriptedLLM({"answerable": False, "answer": "The document does not list ride distances [S1].", "citations": []})
    result = RAGPipeline(embedder, llm, top_k=3, min_relevance=0.05).answer(docs, "How long are the Saturday group rides?")
    assert result.status == "abstained"
    assert result.sources == []
    assert "[S1]" not in result.answer


def test_abstains_when_model_cites_only_unknown_sources(embedder, docs):
    llm = ScriptedLLM({"answerable": True, "answer": "Founded in 1950 [S9].", "citations": ["S9"]})
    result = RAGPipeline(embedder, llm, top_k=3, min_relevance=0.05).answer(docs, "When was the club founded?")
    assert result.status == "abstained"
    assert result.answer == ABSTAIN_UNGROUNDED
    assert result.sources == []


def test_abstains_when_model_gives_uncited_answer(embedder, docs):
    llm = ScriptedLLM({"answerable": True, "answer": "It was founded in 1987.", "citations": []})
    result = RAGPipeline(embedder, llm, top_k=3, min_relevance=0.05).answer(docs, "When was the club founded?")
    assert result.status == "abstained"


def test_retrieval_only_mode_returns_passages_without_answer(embedder, docs):
    result = RAGPipeline(embedder, None, top_k=3, min_relevance=0.05).answer(docs, "When was the club founded?")
    assert result.status == "retrieval_only"
    assert result.sources[0].page == 1
    assert result.model is None


def test_llm_errors_propagate(embedder, docs):
    llm = ScriptedLLM(LLMError("boom"))
    with pytest.raises(LLMError):
        RAGPipeline(embedder, llm, top_k=3, min_relevance=0.05).answer(docs, "When was the club founded?")


# ---- pipeline: streaming mode -----------------------------------------------------------


def _collect(events):
    events = list(events)
    tokens = "".join(d["text"] for e, d in events if e == "token")
    return [e for e, _ in events], tokens, events[-1][1]


def test_stream_emits_sources_tokens_then_validated_final(embedder, docs):
    llm = ScriptedLLM("The club was founded in 1987 [S1]. It meets often [S8].")
    kinds, tokens, final = _collect(RAGPipeline(embedder, llm, 3, 0.05).answer_stream(docs, "When was the club founded?"))
    assert kinds[0] == "sources" and kinds[-1] == "final" and "token" in kinds
    assert tokens == "The club was founded in 1987 [S1]. It meets often [S8]."
    assert final["status"] == "answered"
    assert "[S8]" not in final["answer"]  # streamed draft is corrected by the final event
    assert [s["page"] for s in final["sources"]] == [1]


def test_stream_not_found_is_an_abstention_and_never_streamed(embedder, docs):
    llm = ScriptedLLM("NOT_FOUND: The document does not say how long rides are.")
    kinds, tokens, final = _collect(RAGPipeline(embedder, llm, 3, 0.05).answer_stream(docs, "How long are the rides?"))
    assert tokens == ""
    assert final["status"] == "abstained"
    assert final["answer"] == "The document does not say how long rides are."


def test_stream_uncited_answer_becomes_abstention(embedder, docs):
    _, _, final = _collect(RAGPipeline(embedder, ScriptedLLM("Founded in 1987."), 3, 0.05).answer_stream(docs, "When founded?"))
    assert final["status"] == "abstained"
    assert final["answer"] == ABSTAIN_UNGROUNDED


def test_stream_without_evidence_skips_llm(embedder, docs):
    llm = ScriptedLLM()
    kinds, _, final = _collect(RAGPipeline(embedder, llm, 3, 0.2).answer_stream(docs, "Quantum chromodynamics?"))
    assert kinds == ["final"]
    assert final["status"] == "abstained"
    assert llm.prompts == []


def test_stream_llm_failure_becomes_error_event(embedder, docs):
    llm = ScriptedLLM(LLMError("Gemini is overloaded right now.", code="llm_overloaded"))
    events = list(RAGPipeline(embedder, llm, 3, 0.05).answer_stream(docs, "When was the club founded?"))
    assert events[-1] == ("error", {"code": "llm_overloaded", "message": "Gemini is overloaded right now."})
