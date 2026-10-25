# DocuMind AI

Ask questions about a PDF and get answers grounded in its text, with every claim cited to the page it came from. Select a citation to see the exact passage highlighted on the rendered page. When the document doesn't contain the answer, DocuMind says so instead of guessing.

![A Gemini answer citing page 4, the cited passage highlighted on the page, and a declined question the handbook doesn't answer](docs/screenshots/answer.png)

**Live demo:** https://documind-ai-blush.vercel.app (select "Try the sample library handbook")

**Stack:** Next.js 16 · TypeScript · Tailwind CSS 4 | FastAPI · PyMuPDF · sentence-transformers (all-MiniLM-L6-v2) · FAISS | Gemini API, with a local Ollama fallback | Vercel (frontend, Python function, Blob)

---

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [How the RAG pipeline works](#how-the-rag-pipeline-works)
- [Local setup](#local-setup)
- [Environment variables](#environment-variables)
- [Testing](#testing)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [API reference](#api-reference)
- [Limitations and next steps](#limitations-and-next-steps)

## Features

- **PDF upload with validation.** Checks the extension, size (20 MB), PDF header, encryption, page count (500), and whether there is a text layer. Scanned (image-only) PDFs get a specific message that suggests OCR.
- **Page-preserving extraction.** Text is extracted page by page; chunks never cross a page boundary, so every citation's page number is exact. Repeated headers and footers are stripped so they don't pollute retrieval.
- **Semantic retrieval.** all-MiniLM-L6-v2 embeddings in a per-document FAISS index; each vector id maps explicitly to its source passage.
- **Grounded generation.** The LLM sees only the retrieved passages, labelled `S1..Sn`, and must answer in JSON with inline citations.
- **Citation validation.** Citation ids are checked against the passages actually sent to the model. Unknown ids are removed, and an answer left with no valid citation is withheld.
- **Abstention in two stages.** Clearly off-topic questions are rejected before the LLM is called; in-domain questions the document doesn't answer are declined by the grounding check.
- **Source viewer.** The backend renders the cited page to PNG and highlights the passage on it.
- **Retrieval-only mode.** Without a configured LLM, the app shows the most relevant passages and labels them clearly. It never fabricates an answer.
- **Polished UI.** Document sidebar, upload with progress, chat with loading, empty, error and retry states, a responsive layout down to phone width, keyboard focus states, and reduced-motion support.

| Welcome | Off-topic question declined | Mobile |
| --- | --- | --- |
| ![Welcome screen](docs/screenshots/welcome.png) | ![Abstained answer](docs/screenshots/abstained.png) | ![Mobile layout](docs/screenshots/mobile.png) |

> The hero screenshot is from the live deployment (Vercel, `gemini-3.1-flash-lite`). The three above were captured locally in retrieval-only mode, before a key was configured.

## Architecture

```
┌────────────────────────── Browser ───────────────────────────┐
│ Next.js app (Vercel)                                          │
│  Sidebar · Upload · Conversation · Source panel               │
└───────────────┬───────────────────────────────────────────────┘
                │ HTTPS + JSON / multipart  (NEXT_PUBLIC_API_BASE_URL)
┌───────────────▼─────────────── FastAPI backend ───────────────┐
│ POST /api/documents                                           │
│   validate → PyMuPDF extract (per page) → strip headers       │
│   → page-scoped chunks → MiniLM embeddings → FAISS index      │
│   → persist to data/documents/<id>/                           │
│ POST /api/documents/{id}/ask                                  │
│   embed question → FAISS top-k → relevance floor              │
│   → prompt with S1..Sn → LLM (JSON) → validate citations      │
│ GET  /api/documents/{id}/pages/{n}?chunk_id=…  (PNG + highlight)│
└──────┬──────────────────────────────┬─────────────────────────┘
       │ server-side key only         │ optional, local
┌──────▼──────────┐          ┌────────▼────────┐
│ Gemini API      │          │ Ollama          │
└─────────────────┘          └─────────────────┘
```

The browser calls the backend directly rather than through a Next.js proxy, because Vercel functions cap request bodies at 4.5 MB, which would break 20 MB uploads. The Gemini key exists only in the backend environment.

```
.
├── backend/
│   ├── app/
│   │   ├── main.py            FastAPI app, routes, error envelope
│   │   ├── config.py          settings from env / .env
│   │   ├── pdf_parser.py      validation + page-preserving extraction
│   │   ├── chunker.py         page-scoped, sentence-packed chunks
│   │   ├── embeddings.py      sentence-transformers wrapper
│   │   ├── vector_store.py    FAISS index with explicit id → passage map
│   │   ├── document_store.py  ingestion + on-disk persistence
│   │   ├── rag.py             retrieval, prompt, citation validation, abstention
│   │   ├── llm.py             Gemini (REST) and Ollama clients
│   │   ├── page_render.py     page PNG with highlighted passage
│   │   ├── schemas.py         request/response models
│   │   └── errors.py          domain errors → HTTP codes
│   ├── tests/                 pytest suite (85 tests)
│   ├── scripts/
│   │   ├── make_sample_pdf.py builds the synthetic sample PDF
│   │   └── evaluate.py        retrieval + answer evaluation
│   ├── eval/sample_eval.json  30 labelled questions
│   ├── sample_data/           synthetic sample handbook (7 pages)
│   ├── Dockerfile
│   ├── requirements.txt / requirements-dev.txt
│   └── .env.example
├── frontend/
│   ├── src/app/               layout, page, global styles (design tokens)
│   ├── src/components/        Workspace, Sidebar, UploadDropzone, Message, Composer, SourcePanel
│   ├── src/lib/               API client, citation parsing (+ Vitest tests)
│   └── .env.example
├── docs/screenshots/
└── .github/workflows/ci.yml
```

## How the RAG pipeline works

**Ingestion (once per upload)**

1. **Validate.** The extension is `.pdf`, the file isn't empty, it's under 20 MB, the `%PDF-` header is present, it isn't encrypted, and it has ≤ 500 pages.
2. **Extract.** PyMuPDF reads text blocks for each page in reading order. Line-wrap hyphenation is repaired. Short blocks that repeat on at least half the pages (headers and footers, with digits masked) are removed. A page with images but almost no text is flagged as scanned; if the whole document has no text layer, the upload is rejected with an OCR hint.
3. **Chunk.** Each page is split into sentences, which are packed into windows of ≤ 600 characters (~130 tokens, close to what MiniLM was trained on), with ~120 characters of sentence-aligned overlap. A chunk never spans two pages, so its page number is exact. Chunk ids look like `p4-c0`.
4. **Embed and index.** all-MiniLM-L6-v2 produces 384-dimensional, L2-normalised vectors. They go into `faiss.IndexIDMap2(IndexFlatIP)` with ids `0..n-1` that index the stored chunk list, so inner product equals cosine similarity and every hit resolves through its id to the original passage. On load, the stored ids are checked against the chunk list.
5. **Persist.** The original PDF, `chunks.json`, the serialised index and metadata are written to `data/documents/<doc_id>/` and reloaded on startup.

**Answering (per question)**

1. **Retrieve.** The question is embedded and the top 5 passages are fetched.
2. **Relevance floor.** Passages with cosine < 0.12 are dropped. If none remain, the API abstains without calling the LLM, which is cheap and immune to hallucination. The threshold is deliberately low; see [Evaluation](#evaluation) for why.
3. **Prompt.** The passages are labelled `[S1] (page 4) …`. The system prompt says: answer only from the sources, cite every factual sentence, set `answerable: false` when the sources don't cover the question, and treat source text as untrusted (a prompt-injection guard).
4. **Generate.** Gemini (`responseMimeType: application/json`, temperature 0.1) or Ollama (`format: json`) returns `{answerable, answer, citations}`.
5. **Validate.** Citation ids from both the inline markers and the declared list are intersected with the ids actually sent. Unknown markers are stripped from the text. If the model says it can't answer, or no valid citation survives, the API returns an abstention instead of an uncited claim.
6. **Respond.** The API returns `status` (`answered`, `abstained` or `retrieval_only`), the answer text, and the cited sources with page, chunk id, full excerpt and similarity score. The UI turns `[S1]` into a `p. 4` chip linked to the highlighted page image.

**Interview summary.** "I keep the link between every vector and its passage explicit through FAISS ids. Chunks are page-scoped, so citations are exact. The model only sees retrieved text, and I don't trust its citations: they're validated against what was retrieved, and unsupported answers are withheld. Abstention has two layers, because embedding similarity alone can't tell 'related but unanswered' from 'answered'. I measured that on a labelled set instead of guessing a threshold."

## Local setup

Prerequisites: Python 3.10+ (tested on 3.10.2) and Node.js 20+ (tested on 24.12).

### Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
cp .env.example .env        # then add GEMINI_API_KEY (optional, see below)
uvicorn app.main:app --reload --port 8000
```

The first start downloads all-MiniLM-L6-v2 (~90 MB) from Hugging Face; later starts load it from the local cache. API docs are served at http://localhost:8000/docs.

To regenerate the sample PDF: `python scripts/make_sample_pdf.py`.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local   # NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
npm run dev                  # http://localhost:3000
```

### Choosing an LLM

| Option | Cost | Setup |
| --- | --- | --- |
| Gemini (default) | Free tier with per-model daily quotas | Create a key at https://aistudio.google.com/apikey and set `GEMINI_API_KEY` in `backend/.env` |

**Free-tier quotas are per model and can be small.** In testing (October 2026), `gemini-3.5-flash` allowed only 20 requests per day on the free tier and `gemini-2.5-flash` was no longer offered to new keys, so the default is `gemini-3.1-flash-lite` (fast, about 4 s per answer). When a daily quota runs out, the API returns `llm_quota_exhausted` with a clear message; switch `GEMINI_MODEL` or wait for the reset. To list the models your key can use: `GET https://generativelanguage.googleapis.com/v1beta/models` with the header `x-goog-api-key`. Transient 5xx "overloaded" responses are retried automatically (twice, with backoff).
| Ollama (local) | Free, runs on your machine | Install Ollama, run `ollama pull llama3.2:3b`, and keep `ollama serve` running |
| None | Free | Retrieval-only mode: relevant passages are shown and no answer is generated |

With `LLM_PROVIDER=auto` (the default), the backend uses Gemini if a key is set, otherwise Ollama if it's reachable with the configured model, otherwise retrieval-only. `GET /api/health` and the sidebar footer show which one is active.

## Environment variables

**Backend (`backend/.env`)**

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `auto` | `auto`, `gemini`, `ollama` or `none` |
| `GEMINI_API_KEY` | (empty) | Gemini key. Secret; server-side only |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | Any Gemini model that supports JSON output |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `llama3.2:3b` | Must be pulled first |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-request LLM timeout |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated frontend origins |
| `TOP_K` | `5` | Passages retrieved per question |
| `MIN_RELEVANCE` | `0.12` | Cosine floor for evidence |
| `CHUNK_MAX_CHARS` / `CHUNK_OVERLAP_CHARS` | `600` / `120` | Chunking |
| `MAX_UPLOAD_MB` / `MAX_PAGES` | `20` / `500` | Upload limits |
| `DATA_DIR` | `backend/data` | Where documents and indexes are stored |

**Frontend (`frontend/.env.local`)**

| Variable | Purpose |
| --- | --- |
| `NEXT_PUBLIC_API_BASE_URL` | Backend URL. This value is public: it's inlined into the browser bundle at build time, so it must never hold a secret |

`.env` files are git-ignored; only the `.env.example` files are committed.

## Testing

```bash
cd backend
pytest -m "not model"     # 78 fast offline tests (fake embedder + scripted LLM)
pytest -m model           # 7 tests with the real MiniLM model on the sample PDF
pytest                    # all 85

cd frontend
npm run lint && npm run typecheck && npm test && npm run build
```

What the tests cover:

| Area | Examples |
| --- | --- |
| Parsing | page numbers preserved, blank pages keep position, hyphenation repair, header/footer removal, empty, scanned, partly scanned, encrypted, corrupted, too many pages, upload validation |
| Chunking | never crosses pages, size limit, sentence overlap, over-long sentences, unique ids |
| Vector store | hits resolve to the right passage, save/load round trip, mismatched id map rejected |
| RAG | unknown citation ids stripped, abstains with no evidence (LLM not called), abstains when the model says unanswerable, abstains on uncited or invalidly cited answers, retrieval-only mode, malformed model JSON |
| LLM clients | key sent in a header (never the URL), thought parts ignored, 429/400/403/500/blocked/empty/timeout mapped to clear errors, provider selection |
| API | upload errors (400/413/422), question validation, 404s, LLM failure → 502 envelope, page image + highlight, delete, persistence across restart, sample loading |
| Frontend (Vitest) | citation segmentation drops unknown ids and collapses duplicates, file checks, error-envelope parsing |
| Real model | expected page ranked first for 5 questions; off-topic questions fall below the floor |

The unit and API tests use a hashing embedder and a scripted LLM as test doubles. They exist only in `tests/`; the application never uses them.

## Evaluation

`scripts/evaluate.py` runs the real pipeline against `eval/sample_eval.json`: 22 answerable questions labelled with the page that answers them, and 8 the handbook doesn't answer (4 off-topic, 4 in-domain but not covered).

```bash
cd backend
python scripts/evaluate.py                         # retrieval metrics, no LLM needed
python scripts/evaluate.py --llm --out eval/results.json   # adds answer-level metrics
```

**Retrieval results** (measured on 2026-10-24; one 7-page synthetic document, k = 5, 9 chunks):

| Metric | Value |
| --- | --- |
| hit@1 (top passage is from the right page) | 0.864 (19/22) |
| hit@5 | 1.000 (22/22) |
| MRR | 0.924 |
| Answerable questions kept by the 0.12 floor | 22/22 |
| Unanswerable questions rejected by the floor alone | 4/8 (all 4 off-topic ones) |

What the score distribution showed: off-topic questions scored ≤ 0.08, the weakest answerable one scored 0.17, but in-domain unanswerable questions ("Who is the head librarian?", 0.27–0.44) overlap the answerable range (median 0.49). No similarity threshold separates those, which is why the floor is only a coarse filter and in-domain abstention relies on the LLM's `answerable` flag and citation validation. Removing repeated footers raised hit@1 from 0.818 to 0.864 on this set.

**Answer results** with `gemini-3.1-flash-lite` (measured on 2026-10-24; `--llm --delay 5`; raw output in [`backend/eval/results.json`](backend/eval/results.json)):

| Metric | Value |
| --- | --- |
| Answerable questions answered (not abstained) | 22/22 |
| Answers whose citations all point to an expected page | 22/22 |
| Answers factually correct (manual review against the handbook) | 22/22 |
| Unanswerable questions declined | 7/8 |
| Errors | 0 |

- The model also recovered the 3 retrieval cases where the right passage ranked 2nd or 3rd, citing the correct one.
- The one miss, "Can I bring my pet dog?", is a grounded partial answer: the model replied that service animals are welcome but the handbook doesn't say pets are allowed, citing page 7. Against the strict label it counts as a miss.
- The 4 in-domain unanswerables that passed the retrieval floor (head librarian, budget, passport services, pets) are where the LLM's grounding check matters. It declined 3 of the 4.

Correctness was reviewed by hand, not auto-scored.

**Caveats:** 30 questions on one small synthetic document is a smoke test, not a benchmark. To evaluate properly, add 2–3 real documents (a paper, a policy, a manual) with 30+ labelled questions each, and re-tune `MIN_RELEVANCE` with the sweep the script prints.

## Deployment

Both halves run on Vercel's free Hobby plan as two projects linked to the same repository. Live: frontend https://documind-ai-blush.vercel.app, API https://documind-api-three.vercel.app/api/health.

### 1. Backend on Vercel (Python function)

Vercel detects the FastAPI `app` in `backend/app/main.py`. `backend/vercel.json` bakes the embedding model into the build (`scripts/fetch_model.py`) and allows 300-second requests; `backend/.python-version` selects Python 3.12, and `requirements.txt` installs CPU-only PyTorch from a wheel URL on Linux.

1. Create a project from the repository with **Root Directory** `backend`.
2. Environment variables:
   - `GEMINI_API_KEY` (mark **Sensitive**)
   - `GEMINI_MODEL=gemini-3.1-flash-lite`
   - `CORS_ORIGINS=https://<your-frontend>.vercel.app`
   - `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`: torch and sentence-transformers make a ~1.3 GB bundle, above the 500 MB standard limit; this enables large functions (up to 5 GB, beta)
3. **Storage → Create → Blob**, access **Private**, connected to the project. This adds `BLOB_READ_WRITE_TOKEN`. Serverless instances don't share a disk, so each document's PDF, chunks, index and metadata are mirrored to Blob (`app/blob_mirror.py`), and any instance can load any document.
4. Deploy, then check `/api/health`: it should report `"llm_provider": "gemini"` and `"storage": "blob"`.

Platform limits that shape the design:

- **Upload size:** Vercel caps request bodies at 4.5 MB, so uploads are limited to 4 MB on Vercel (`MAX_UPLOAD_MB`). `/api/health` reports the limit and the frontend enforces it. Direct-to-Blob client uploads would lift this; it's listed under next steps.
- **Cold starts:** the first request after a period of inactivity loads PyTorch and the model, which takes several seconds.
- **Writable disk:** only `/tmp` is writable (`DATA_DIR` defaults to `/tmp/documind` on Vercel), and it is only a cache in front of Blob.

Other hosts: `backend/Dockerfile` (Python 3.12, port 7860, model baked in) runs on any container host with about 1 GB of RAM. Hugging Face now requires a paid plan for Docker and Gradio Spaces; `deploy/hf-space/` keeps a tested Gradio-Space entry point for accounts that have one.

### 2. Frontend on Vercel

1. Create a second project from the same repository with **Root Directory** `frontend`.
2. Set `NEXT_PUBLIC_API_BASE_URL=https://<your-backend>.vercel.app`. It is inlined at build time, so redeploy after changing it.
3. Deploy.

New Vercel projects may have **Deployment Protection** (Vercel Authentication) enabled. For a public demo, set it to cover preview deployments only, under **Settings → Deployment Protection**.

### Before you share the link

- `GET /api/health` reports the expected LLM provider.
- Load the sample, ask one answerable and one unanswerable question, and open a citation.
- Note that there is no authentication: every visitor shares the same document list. See below.

## API reference

All errors use one envelope: `{"error": {"code": "...", "message": "..."}}`.

| Method & path | Description |
| --- | --- |
| `GET /api/health` | Status, active LLM provider and model, embedding model, document count |
| `GET /api/documents` | List documents |
| `POST /api/documents` | Upload (multipart field `file`). `201` with metadata; `400` invalid file, `413` too large, `422` scanned, empty, encrypted or corrupted |
| `POST /api/documents/sample` | Load the bundled sample (idempotent) |
| `GET /api/documents/{id}` | Document metadata |
| `DELETE /api/documents/{id}` | Delete a document and its index |
| `POST /api/documents/{id}/ask` | `{"question": "..."}` → `{status, answer, sources[], model, retrieved}`. `502`/`503` with a clear code when the LLM fails |
| `GET /api/documents/{id}/pages/{n}?chunk_id=p4-c0` | PNG of page *n*, with that passage highlighted |

## Limitations and next steps

Intentionally out of scope for v1:

- **No auth or multi-tenancy.** Documents are global to the server. Next step: per-user sessions, or a document token returned on upload.
- **Single-turn questions.** Each question is answered on its own (follow-ups like "what about DVDs?" aren't rewritten using chat history). Next step: query rewriting from the last few turns.
- **No OCR.** Scanned PDFs are detected and rejected with guidance. Next step: optional `ocrmypdf`/Tesseract.
- **Dense retrieval only.** Exact tokens (phone numbers, codes) are MiniLM's weak spot; 2 of the 3 hit@1 misses were of this kind. Next step: hybrid BM25 + dense retrieval with reciprocal-rank fusion, or a cross-encoder re-ranker.
- **Synchronous ingestion.** Uploads are parsed and embedded inside the request, which is fine for a few hundred pages. Next step: a background job queue with progress reporting for large files.
- **4 MB uploads on Vercel.** This is the platform's request-size cap. Next step: browser-to-Blob client uploads, with the API ingesting from the Blob URL.
- **Not verified in this build:** the Docker image (Docker wasn't installed on the build machine) and the Ollama path (not installed); both are covered by mocked tests only. Gemini and the Vercel deployment were verified live.
