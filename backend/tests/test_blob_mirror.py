"""Serverless behaviour: instances with separate disks sharing one document mirror."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings, _platform_defaults
from app.embeddings import resolve_model_path
from app.main import create_app


class InMemoryMirror:
    """Stands in for Vercel Blob in tests."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def upload(self, doc_id, files):
        for name, body in files.items():
            self.objects[f"{doc_id}/{name}"] = body

    def download(self, doc_id, dest: Path):
        names = [k.split("/", 1)[1] for k in self.objects if k.startswith(f"{doc_id}/")]
        if not names:
            return False
        dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            (dest / name).write_bytes(self.objects[f"{doc_id}/{name}"])
        return True

    def read_meta_files(self):
        return {k.split("/", 1)[0]: v for k, v in self.objects.items() if k.endswith("/meta.json")}

    def delete(self, doc_id):
        for key in [k for k in self.objects if k.startswith(f"{doc_id}/")]:
            del self.objects[key]


def _instance(tmp_path: Path, name: str, embedder, mirror) -> TestClient:
    settings = Settings(_env_file=None, data_dir=tmp_path / name, llm_provider="none", min_relevance=0.05)
    client = TestClient(create_app(settings, embedder=embedder, llm=None, warm_embedder=False, mirror=mirror))
    client.__enter__()
    return client


def test_document_uploaded_on_one_instance_is_usable_on_another(tmp_path, embedder, handbook_pdf):
    mirror = InMemoryMirror()
    a = _instance(tmp_path, "a", embedder, mirror)
    b = _instance(tmp_path, "b", embedder, mirror)

    doc_id = a.post("/api/documents", files={"file": ("h.pdf", handbook_pdf, "application/pdf")}).json()["doc_id"]
    assert set(mirror.objects) == {f"{doc_id}/{n}" for n in ("source.pdf", "chunks.json", "index.faiss", "meta.json")}

    assert [d["doc_id"] for d in b.get("/api/documents").json()] == [doc_id]
    body = b.post(f"/api/documents/{doc_id}/ask", json={"question": "Are helmets mandatory on rides?"}).json()
    assert body["sources"][0]["page"] == 3
    assert b.get(f"/api/documents/{doc_id}/pages/3", params={"chunk_id": "p3-c0"}).status_code == 200

    assert b.delete(f"/api/documents/{doc_id}").status_code == 204
    assert mirror.objects == {}
    assert b.get("/api/documents").json() == []


def test_health_reports_storage_and_upload_limit(tmp_path, embedder):
    blob = _instance(tmp_path, "blob", embedder, InMemoryMirror()).get("/api/health").json()
    local = _instance(tmp_path, "local", embedder, None).get("/api/health").json()
    assert blob["storage"] == "blob" and local["storage"] == "local"
    assert local["max_upload_mb"] == 20


def test_unknown_document_is_404_with_mirror(tmp_path, embedder):
    client = _instance(tmp_path, "x", embedder, InMemoryMirror())
    assert client.get("/api/documents/0123456789ab").status_code == 404


def test_vercel_defaults(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("MAX_UPLOAD_MB", raising=False)
    s = _platform_defaults(Settings(_env_file=None))
    assert s.data_dir == Path("/tmp/documind")
    assert s.max_upload_mb == 4

    monkeypatch.setenv("MAX_UPLOAD_MB", "3")
    assert _platform_defaults(Settings(_env_file=None)).max_upload_mb == 3

    monkeypatch.delenv("VERCEL")
    assert _platform_defaults(Settings(_env_file=None)).max_upload_mb == 3


def test_blob_enabled_by_store_id_or_token():
    assert not Settings(_env_file=None).blob_enabled
    assert Settings(_env_file=None, blob_store_id="store_abc").blob_enabled
    assert Settings(_env_file=None, blob_read_write_token="vercel_blob_rw_x").blob_enabled


def test_resolve_model_path_prefers_baked_copy(tmp_path):
    name = "sentence-transformers/all-MiniLM-L6-v2"
    assert resolve_model_path(name, tmp_path) == name
    (tmp_path / "all-MiniLM-L6-v2").mkdir()
    (tmp_path / "all-MiniLM-L6-v2" / "config.json").write_text("{}")
    assert resolve_model_path(name, tmp_path) == str(tmp_path / "all-MiniLM-L6-v2")
