import pytest

from app.chunker import Chunk
from app.errors import LLMError
from app.rag import (
    ABSTAIN_NO_EVIDENCE,
    ABSTAIN_UNGROUNDED,
    RAGPipeline,
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


@pytest.fixture
def index(embedder):
    return DocumentIndex.build(CHUNKS, embedder.embed([c.text for c in CHUNKS]))


def _sources(*pages: int) -> list[Source]:
    return [Source(f"S{i}", f"p{p}-c0", p, f"text {p}", 0.9) for i, p in enumerate(pages, start=1)]


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


# ---- model output parsing --------------------------------------------------------------


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


# ---- pipeline behaviour ----------------------------------------------------------------


def test_answer_returns_validated_citations_with_pages(embedder, index):
    llm = ScriptedLLM({"answerable": True, "answer": "It costs 40 dollars a year [S1].", "citations": ["S1"]})
    result = RAGPipeline(embedder, llm, top_k=3, min_relevance=0.1).answer(index, "How much does membership cost per year?")
    assert result.status == "answered"
    assert result.sources[0].page == 2
    assert "40 dollars" in result.sources[0].excerpt
    assert result.model == "scripted:test"


def test_abstains_without_calling_llm_when_nothing_is_relevant(embedder, index):
    llm = ScriptedLLM()
    result = RAGPipeline(embedder, llm, top_k=3, min_relevance=0.2).answer(index, "Quantum chromodynamics lattice gauge theory?")
    assert result.status == "abstained"
    assert result.answer == ABSTAIN_NO_EVIDENCE
    assert result.sources == []
    assert llm.prompts == []


def test_abstains_when_model_says_unanswerable(embedder, index):
    llm = ScriptedLLM({"answerable": False, "answer": "The document does not list ride distances [S1].", "citations": []})
    result = RAGPipeline(embedder, llm, top_k=3, min_relevance=0.05).answer(index, "How long are the Saturday group rides?")
    assert result.status == "abstained"
    assert result.sources == []
    assert "[S1]" not in result.answer


def test_abstains_when_model_cites_only_unknown_sources(embedder, index):
    llm = ScriptedLLM({"answerable": True, "answer": "Founded in 1950 [S9].", "citations": ["S9"]})
    result = RAGPipeline(embedder, llm, top_k=3, min_relevance=0.05).answer(index, "When was the club founded?")
    assert result.status == "abstained"
    assert result.answer == ABSTAIN_UNGROUNDED
    assert result.sources == []


def test_abstains_when_model_gives_uncited_answer(embedder, index):
    llm = ScriptedLLM({"answerable": True, "answer": "It was founded in 1987.", "citations": []})
    result = RAGPipeline(embedder, llm, top_k=3, min_relevance=0.05).answer(index, "When was the club founded?")
    assert result.status == "abstained"


def test_retrieval_only_mode_returns_passages_without_answer(embedder, index):
    result = RAGPipeline(embedder, None, top_k=3, min_relevance=0.05).answer(index, "When was the club founded?")
    assert result.status == "retrieval_only"
    assert result.sources[0].page == 1
    assert result.model is None


def test_llm_errors_propagate(embedder, index):
    llm = ScriptedLLM(LLMError("boom"))
    with pytest.raises(LLMError):
        RAGPipeline(embedder, llm, top_k=3, min_relevance=0.05).answer(index, "When was the club founded?")
