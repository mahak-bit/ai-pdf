from app.errors import LLMError

from conftest import ScriptedLLM, make_pdf, make_scanned_pdf


def _upload(client, data: bytes, name: str = "handbook.pdf", mime: str = "application/pdf"):
    return client.post("/api/documents", files={"file": (name, data, mime)})


def test_health_reports_retrieval_only_without_llm(make_client):
    body = make_client().get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "none"
    assert body["documents"] == 0


def test_upload_returns_document_metadata(make_client, handbook_pdf):
    response = _upload(make_client(), handbook_pdf)
    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "handbook.pdf"
    assert body["page_count"] == 3
    assert body["chunk_count"] >= 3
    assert len(body["doc_id"]) == 12


def test_upload_rejects_non_pdf(make_client):
    response = _upload(make_client(), b"hello", name="notes.txt", mime="text/plain")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_upload"


def test_upload_rejects_renamed_non_pdf(make_client):
    response = _upload(make_client(), b"\x89PNG\r\n\x1a\nnot a pdf", name="photo.pdf")
    assert response.status_code == 400
    assert "not a valid PDF" in response.json()["error"]["message"]


def test_upload_rejects_empty_file(make_client):
    response = _upload(make_client(), b"")
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "The uploaded file is empty."


def test_upload_rejects_oversized_file(make_client):
    response = _upload(make_client(), b"%PDF-1.7" + b"0" * (2 * 1024 * 1024))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_upload_rejects_scanned_pdf(make_client):
    response = _upload(make_client(), make_scanned_pdf())
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "scanned_pdf"


def test_upload_rejects_pdf_without_text(make_client):
    response = _upload(make_client(), make_pdf(["", ""]))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "no_text"


def test_upload_without_file_field(make_client):
    response = make_client().post("/api/documents")
    assert response.status_code == 422
    assert "No file was uploaded" in response.json()["error"]["message"]


def test_ask_returns_cited_answer_with_page(make_client, handbook_pdf):
    llm = ScriptedLLM()
    client = make_client(llm=llm)
    doc_id = _upload(client, handbook_pdf).json()["doc_id"]

    response = client.post(f"/api/documents/{doc_id}/ask", json={"question": "Are helmets mandatory on group rides?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["sources"][0]["page"] == 3
    assert "Helmets are mandatory" in body["sources"][0]["excerpt"]
    assert "(page 3)" in llm.prompts[0]


def test_ask_abstains_on_unrelated_question(make_client, handbook_pdf):
    llm = ScriptedLLM()
    client = make_client(llm=llm)
    doc_id = _upload(client, handbook_pdf).json()["doc_id"]
    body = client.post(f"/api/documents/{doc_id}/ask", json={"question": "Explain photosynthesis in algae"}).json()
    assert body["status"] == "abstained"
    assert body["sources"] == []
    assert llm.prompts == []


def test_ask_in_retrieval_only_mode(make_client, handbook_pdf):
    client = make_client(llm=None)
    doc_id = _upload(client, handbook_pdf).json()["doc_id"]
    body = client.post(f"/api/documents/{doc_id}/ask", json={"question": "How much does membership cost?"}).json()
    assert body["status"] == "retrieval_only"
    assert body["sources"][0]["page"] == 2


def test_ask_validates_question(make_client, handbook_pdf):
    client = make_client()
    doc_id = _upload(client, handbook_pdf).json()["doc_id"]
    for payload in ({"question": "   "}, {"question": ""}, {"question": "x" * 1001}, {}):
        response = client.post(f"/api/documents/{doc_id}/ask", json=payload)
        assert response.status_code == 422, payload
        assert response.json()["error"]["code"] == "validation_error"


def test_ask_unknown_document_is_404(make_client):
    response = make_client().post("/api/documents/0123456789ab/ask", json={"question": "Hello?"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_malformed_document_id_is_rejected(make_client):
    response = make_client().get("/api/documents/..%2F..%2Fetc")
    assert response.status_code in (404, 422)
    assert "error" in response.json()


def test_llm_failure_returns_clear_502(make_client, handbook_pdf):
    client = make_client(llm=ScriptedLLM(LLMError("Gemini took too long to respond.", code="llm_timeout")))
    doc_id = _upload(client, handbook_pdf).json()["doc_id"]
    response = client.post(f"/api/documents/{doc_id}/ask", json={"question": "When was the club founded?"})
    assert response.status_code == 502
    assert response.json()["error"] == {"code": "llm_timeout", "message": "Gemini took too long to respond."}


def test_malformed_llm_output_returns_502(make_client, handbook_pdf):
    client = make_client(llm=ScriptedLLM("not json at all"))
    doc_id = _upload(client, handbook_pdf).json()["doc_id"]
    response = client.post(f"/api/documents/{doc_id}/ask", json={"question": "When was the club founded?"})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "llm_bad_output"


def test_page_image_with_highlight(make_client, handbook_pdf):
    client = make_client()
    doc_id = _upload(client, handbook_pdf).json()["doc_id"]
    response = client.get(f"/api/documents/{doc_id}/pages/2", params={"chunk_id": "p2-c0"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_page_image_rejects_chunk_from_other_page(make_client, handbook_pdf):
    client = make_client()
    doc_id = _upload(client, handbook_pdf).json()["doc_id"]
    assert client.get(f"/api/documents/{doc_id}/pages/1", params={"chunk_id": "p2-c0"}).status_code == 404
    assert client.get(f"/api/documents/{doc_id}/pages/99").status_code == 404


def test_list_and_delete_documents(make_client, handbook_pdf):
    client = make_client()
    doc_id = _upload(client, handbook_pdf).json()["doc_id"]
    assert [d["doc_id"] for d in client.get("/api/documents").json()] == [doc_id]
    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.get(f"/api/documents/{doc_id}").status_code == 404
    assert client.get("/api/documents").json() == []


def test_documents_survive_restart(make_client, handbook_pdf):
    doc_id = _upload(make_client(), handbook_pdf).json()["doc_id"]
    restarted = make_client()
    assert restarted.get(f"/api/documents/{doc_id}").status_code == 200
    body = restarted.post(f"/api/documents/{doc_id}/ask", json={"question": "Where do group rides leave from?"}).json()
    assert body["sources"][0]["page"] == 3


def test_sample_document_loads_once(make_client):
    client = make_client()
    first = client.post("/api/documents/sample")
    second = client.post("/api/documents/sample")
    assert first.status_code == 201
    assert first.json()["doc_id"] == second.json()["doc_id"]
    assert first.json()["page_count"] == 7


# ---- multiple documents and streaming -----------------------------------------------------


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    import json

    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


ROWING = make_pdf(["The rowing club charges 90 dollars per year.", "Boats are kept in the boathouse by the river."])


def test_ask_across_several_documents_names_each_source(make_client, handbook_pdf):
    llm = ScriptedLLM({"answerable": True, "answer": "Cycling is 40 dollars [S1]; rowing is 90 [S2].", "citations": ["S1", "S2"]})
    client = make_client(llm=llm, top_k=4, min_relevance=0.05)
    cycling = _upload(client, handbook_pdf, name="cycling.pdf").json()["doc_id"]
    rowing = _upload(client, ROWING, name="rowing.pdf").json()["doc_id"]

    body = client.post("/api/ask", json={"doc_ids": [cycling, rowing], "question": "How much is yearly membership?"}).json()
    assert body["status"] == "answered"
    assert {s["filename"] for s in body["sources"]} <= {"cycling.pdf", "rowing.pdf"}
    assert all(s["doc_id"] in (cycling, rowing) for s in body["sources"])
    assert "cycling.pdf, page" in llm.prompts[0] and "rowing.pdf, page" in llm.prompts[0]


def test_ask_validates_document_list(make_client, handbook_pdf):
    client = make_client()
    doc_id = _upload(client, handbook_pdf).json()["doc_id"]
    assert client.post("/api/ask", json={"doc_ids": [], "question": "Hi?"}).status_code == 422
    assert client.post("/api/ask", json={"doc_ids": ["nope"], "question": "Hi?"}).status_code == 422
    assert client.post("/api/ask", json={"doc_ids": [doc_id, "0123456789ab"], "question": "Hi?"}).status_code == 404


def test_stream_endpoint_sends_sources_tokens_and_final(make_client, handbook_pdf):
    client = make_client(llm=ScriptedLLM("Helmets are mandatory on every ride [S1]."))
    doc_id = _upload(client, handbook_pdf).json()["doc_id"]
    response = client.post("/api/ask/stream", json={"doc_ids": [doc_id], "question": "Are helmets mandatory on rides?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "final" and kinds.count("token") > 1
    final = events[-1][1]
    assert final["status"] == "answered"
    assert final["sources"][0]["page"] == 3


def test_stream_endpoint_reports_llm_failure_as_event(make_client, handbook_pdf):
    client = make_client(llm=ScriptedLLM(LLMError("Gemini is overloaded right now.", code="llm_overloaded")))
    doc_id = _upload(client, handbook_pdf).json()["doc_id"]
    events = _parse_sse(client.post("/api/ask/stream", json={"doc_ids": [doc_id], "question": "When was the club founded?"}).text)
    assert events[-1] == ("error", {"code": "llm_overloaded", "message": "Gemini is overloaded right now."})


def test_stream_endpoint_unknown_document_is_plain_404(make_client):
    response = make_client().post("/api/ask/stream", json={"doc_ids": ["0123456789ab"], "question": "Hi?"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
