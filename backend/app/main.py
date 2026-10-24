"""FastAPI application: routes, error handling and wiring."""

import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Path as PathParam, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import BACKEND_ROOT, Settings, get_settings
from .blob_mirror import DocumentMirror, VercelBlobMirror
from .document_store import DocumentStore
from .embeddings import Embedder, SentenceTransformerEmbedder, resolve_model_path
from .errors import AppError, NotFound
from .llm import LLMClient, build_llm_client
from .page_render import render_page_png
from .pdf_parser import validate_upload
from .rag import RAGPipeline
from .schemas import AnswerOut, AskRequest, DocumentOut, ErrorOut, HealthOut

log = logging.getLogger("documind")

SAMPLE_PDF = BACKEND_ROOT / "sample_data" / "harbor-city-library-handbook.pdf"
MODELS_DIR = BACKEND_ROOT / "models"
DocId = Annotated[str, PathParam(pattern=r"^[0-9a-f]{12}$", description="Document id")]
_UNSET = object()


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    llm: LLMClient | None | object = _UNSET,
    warm_embedder: bool = True,
    mirror: DocumentMirror | None | object = _UNSET,
) -> FastAPI:
    settings = settings or get_settings()
    embedder = embedder or SentenceTransformerEmbedder(resolve_model_path(settings.embedding_model, MODELS_DIR))
    if mirror is _UNSET:
        mirror = VercelBlobMirror(settings.blob_access) if settings.blob_read_write_token else None
    llm_client: LLMClient | None = build_llm_client(settings) if llm is _UNSET else llm  # type: ignore[assignment]

    store = DocumentStore(
        settings.data_dir, embedder, settings.chunk_max_chars, settings.chunk_overlap_chars, mirror=mirror  # type: ignore[arg-type]
    )
    pipeline = RAGPipeline(embedder, llm_client, settings.top_k, settings.min_relevance)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loaded = store.load_all()
        log.info("Loaded %d stored document(s); LLM: %s", loaded, llm_client.name if llm_client else "none")
        if warm_embedder:
            # Load the embedding model in the background so the first upload is faster.
            threading.Thread(target=lambda: embedder.embed(["warm up"]), daemon=True).start()
        yield

    app = FastAPI(
        title="DocuMind AI",
        version="1.0.0",
        description="Ask questions about a PDF and get answers cited to the page.",
        lifespan=lifespan,
        responses={400: {"model": ErrorOut}, 404: {"model": ErrorOut}, 422: {"model": ErrorOut}},
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    # ---- error handling -------------------------------------------------------------

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError):
        return _error(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = [str(p) for p in first.get("loc", []) if p not in ("body", "path", "query")]
        if loc == ["file"]:
            message = "No file was uploaded. Attach a PDF in the 'file' form field."
        else:
            field = loc[-1] if loc else "request"
            message = f"Invalid {field}: {first.get('msg', 'invalid value')}".replace("Value error, ", "")
        return _error(422, "validation_error", message)

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(_: Request, exc: StarletteHTTPException):
        return _error(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def unhandled_handler(_: Request, exc: Exception):
        log.exception("Unhandled error", exc_info=exc)
        return _error(500, "internal_error", "Something went wrong on the server. Try again.")

    # ---- routes ---------------------------------------------------------------------

    @app.get("/api/health", response_model=HealthOut)
    def health():
        return HealthOut(
            status="ok",
            llm_provider=llm_client.name if llm_client else "none",
            llm_model=llm_client.model if llm_client else None,
            embedding_model=settings.embedding_model,
            embedding_model_loaded=bool(getattr(embedder, "loaded", True)),
            documents=len(store.list()),
            max_upload_mb=settings.max_upload_mb,
            storage="blob" if store.mirror else "local",
        )

    @app.get("/api/documents", response_model=list[DocumentOut])
    def list_documents():
        return [asdict(m) for m in store.list()]

    # Sync handlers run in FastAPI's threadpool, so parsing and embedding don't block the loop.
    @app.post("/api/documents", response_model=DocumentOut, status_code=201)
    def upload_document(file: UploadFile = File(...)):
        max_bytes = settings.max_upload_mb * 1024 * 1024
        data = file.file.read(max_bytes + 1)
        validate_upload(file.filename, data, max_bytes)
        return asdict(store.ingest(file.filename or "document.pdf", data, settings.max_pages))

    @app.post("/api/documents/sample", response_model=DocumentOut, status_code=201)
    def load_sample():
        existing = next((m for m in store.list() if m.filename == SAMPLE_PDF.name), None)
        if existing:
            return asdict(existing)
        if not SAMPLE_PDF.exists():
            raise NotFound("The sample document is missing on the server. Run scripts/make_sample_pdf.py.")
        return asdict(store.ingest(SAMPLE_PDF.name, SAMPLE_PDF.read_bytes(), settings.max_pages))

    @app.get("/api/documents/{doc_id}", response_model=DocumentOut)
    def get_document(doc_id: DocId):
        return asdict(store.get(doc_id).meta)

    @app.delete("/api/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: DocId):
        store.delete(doc_id)
        return Response(status_code=204)

    @app.post("/api/documents/{doc_id}/ask", response_model=AnswerOut)
    def ask(doc_id: DocId, body: AskRequest):
        doc = store.get(doc_id)
        return asdict(pipeline.answer(doc.index, body.question))

    @app.get("/api/documents/{doc_id}/pages/{page}", response_class=Response)
    def page_image(
        doc_id: DocId,
        page: Annotated[int, PathParam(ge=1)],
        chunk_id: Annotated[str | None, Query(pattern=r"^p\d+-c\d+$")] = None,
    ):
        doc = store.get(doc_id)
        highlight = None
        if chunk_id:
            chunk = next((c for c in doc.index.chunks if c.chunk_id == chunk_id), None)
            if chunk is None or chunk.page != page:
                raise NotFound("That passage does not belong to this page.")
            highlight = chunk.text
        png = render_page_png(Path(doc.pdf_path), page, highlight)
        return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})

    app.state.store = store
    app.state.pipeline = pipeline
    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
