"""Ingestion and on-disk persistence of documents.

Layout per document::

    data/documents/<doc_id>/
        source.pdf    original upload (used to render cited pages)
        chunks.json   passages, in vector-id order
        index.faiss   serialised FAISS index
        meta.json     filename, page count, timestamps
"""

import json
import logging
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .chunker import Chunk, chunk_pages
from .embeddings import Embedder
from .errors import NotFound, UnprocessableDocument
from .pdf_parser import extract_pages
from .vector_store import DocumentIndex

log = logging.getLogger(__name__)


@dataclass
class DocumentMeta:
    doc_id: str
    filename: str
    page_count: int
    chunk_count: int
    char_count: int
    created_at: str
    title: str | None = None
    image_only_pages: list[int] = field(default_factory=list)


@dataclass
class StoredDocument:
    meta: DocumentMeta
    index: DocumentIndex
    pdf_path: Path


class DocumentStore:
    def __init__(self, root: Path, embedder: Embedder, chunk_max_chars: int, chunk_overlap_chars: int):
        self.root = root / "documents"
        self.root.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.chunk_max_chars = chunk_max_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self._docs: dict[str, StoredDocument] = {}
        self._lock = threading.Lock()

    # ---- ingestion -------------------------------------------------------------------

    def ingest(self, filename: str, data: bytes, max_pages: int) -> DocumentMeta:
        parsed = extract_pages(data, max_pages=max_pages)
        chunks = chunk_pages(parsed.pages, self.chunk_max_chars, self.chunk_overlap_chars)
        if not chunks:
            raise UnprocessableDocument("No readable text was found in this PDF.", code="no_text")

        vectors = self.embedder.embed([c.text for c in chunks])
        index = DocumentIndex.build(chunks, vectors)

        meta = DocumentMeta(
            doc_id=uuid.uuid4().hex[:12],
            filename=Path(filename).name,
            page_count=parsed.page_count,
            chunk_count=len(chunks),
            char_count=parsed.char_count,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            title=parsed.title,
            image_only_pages=parsed.image_only_pages,
        )
        folder = self.root / meta.doc_id
        folder.mkdir(parents=True)
        try:
            (folder / "source.pdf").write_bytes(data)
            (folder / "chunks.json").write_text(json.dumps([c.to_dict() for c in chunks]), encoding="utf-8")
            index.save(folder / "index.faiss")
            (folder / "meta.json").write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise

        with self._lock:
            self._docs[meta.doc_id] = StoredDocument(meta, index, folder / "source.pdf")
        log.info("Ingested %s: %d pages, %d chunks", meta.filename, meta.page_count, meta.chunk_count)
        return meta

    # ---- lookup ----------------------------------------------------------------------

    def load_all(self) -> int:
        """Load previously ingested documents from disk. Skips (and logs) broken folders."""
        loaded = 0
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                continue
            try:
                meta = DocumentMeta(**json.loads((folder / "meta.json").read_text(encoding="utf-8")))
                chunks = [Chunk(**c) for c in json.loads((folder / "chunks.json").read_text(encoding="utf-8"))]
                index = DocumentIndex.load(folder / "index.faiss", chunks)
            except Exception:
                log.exception("Skipping unreadable document folder %s", folder.name)
                continue
            with self._lock:
                self._docs[meta.doc_id] = StoredDocument(meta, index, folder / "source.pdf")
            loaded += 1
        return loaded

    def get(self, doc_id: str) -> StoredDocument:
        with self._lock:
            doc = self._docs.get(doc_id)
        if doc is None:
            raise NotFound("That document was not found. It may have been deleted; upload it again.")
        return doc

    def list(self) -> list[DocumentMeta]:
        with self._lock:
            metas = [d.meta for d in self._docs.values()]
        return sorted(metas, key=lambda m: m.created_at, reverse=True)

    def delete(self, doc_id: str) -> None:
        self.get(doc_id)
        with self._lock:
            self._docs.pop(doc_id, None)
        shutil.rmtree(self.root / doc_id, ignore_errors=True)
