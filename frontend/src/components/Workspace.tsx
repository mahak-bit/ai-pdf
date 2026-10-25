"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, checkFile, DEFAULT_MAX_UPLOAD_MB, type DocumentInfo, type Health, type Source } from "@/lib/api";
import { Composer } from "./Composer";
import { AlertIcon, LogoMark, MenuIcon } from "./icons";
import { Message, type Turn } from "./Message";
import { Sidebar } from "./Sidebar";
import { SourcePanel } from "./SourcePanel";
import { UploadDropzone, type UploadState } from "./UploadDropzone";

const SAMPLE_FILENAME = "harbor-city-library-handbook.pdf";
const SAMPLE_QUESTIONS = [
  "Are there late fees for overdue books?",
  "How far ahead can I book a study room?",
  "Who is the head librarian?",
];
const MAX_SELECTED = 10;

const message = (err: unknown) => (err instanceof ApiError ? err.message : "Something went wrong. Try again.");
/** Conversations are kept per set of selected documents. */
const threadKey = (ids: string[]) => [...ids].sort().join(",");

export function Workspace() {
  const [health, setHealth] = useState<Health | null>(null);
  const [serverDown, setServerDown] = useState(false);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [docsLoading, setDocsLoading] = useState(true);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [threads, setThreads] = useState<Record<string, Turn[]>>({});
  const [openSource, setOpenSource] = useState<Source | null>(null);
  const [upload, setUpload] = useState<UploadState>({ phase: "idle" });
  const [sampleLoading, setSampleLoading] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const scrollEnd = useRef<HTMLDivElement>(null);

  const selectedDocs = documents.filter((d) => selectedIds.includes(d.doc_id));
  const primaryDoc = selectedDocs[0] ?? null;
  const multiDoc = selectedDocs.length > 1;
  const key = threadKey(selectedDocs.map((d) => d.doc_id));
  const turns = selectedDocs.length ? (threads[key] ?? []) : [];
  const asking = turns.some((t) => t.state === "pending");
  const lastDraftLength = turns.at(-1)?.draft?.length ?? 0;
  const maxUploadMb = health?.max_upload_mb ?? DEFAULT_MAX_UPLOAD_MB;
  const sourceDoc = openSource ? (documents.find((d) => d.doc_id === openSource.doc_id) ?? null) : null;

  const connect = useCallback(async () => {
    setDocsLoading(true);
    try {
      const [h, docs] = await Promise.all([api.health(), api.listDocuments()]);
      setHealth(h);
      setServerDown(false);
      setDocuments(docs);
      setSelectedIds((current) => (current.length ? current : docs[0] ? [docs[0].doc_id] : []));
    } catch {
      setServerDown(true);
    } finally {
      setDocsLoading(false);
    }
  }, []);

  useEffect(() => {
    // Initial load from the backend; state is set once the requests resolve.
    void connect();
  }, [connect]);

  useEffect(() => {
    scrollEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length, asking, lastDraftLength]);

  const selectOnly = (docId: string) => {
    setSelectedIds([docId]);
    setOpenSource(null);
    setDrawerOpen(false);
  };

  const toggleSelected = (docId: string) => {
    setSelectedIds((ids) => {
      if (ids.includes(docId)) return ids.length > 1 ? ids.filter((id) => id !== docId) : ids;
      return ids.length >= MAX_SELECTED ? ids : [...ids, docId];
    });
    setOpenSource(null);
  };

  const addDocument = (doc: DocumentInfo) => {
    setDocuments((docs) => [doc, ...docs.filter((d) => d.doc_id !== doc.doc_id)]);
    setServerDown(false);
    setUpload({ phase: "idle" });
    selectOnly(doc.doc_id);
  };

  const handleUpload = async (file: File) => {
    const problem = checkFile(file, maxUploadMb);
    if (problem) return setUpload({ phase: "idle", error: problem });
    setUpload({ phase: "uploading", filename: file.name, progress: 0 });
    const onProgress = (progress: number) =>
      setUpload(progress >= 1 ? { phase: "indexing", filename: file.name } : { phase: "uploading", filename: file.name, progress });
    try {
      const doc = health?.direct_upload ? await api.uploadDirect(file, onProgress) : await api.uploadDocument(file, onProgress);
      addDocument(doc);
    } catch (err) {
      setUpload({ phase: "idle", error: message(err) });
    }
  };

  const handleSample = async () => {
    setSampleLoading(true);
    setNotice(null);
    try {
      addDocument(await api.loadSample());
    } catch (err) {
      setNotice(message(err));
    } finally {
      setSampleLoading(false);
    }
  };

  const handleDelete = async (docId: string) => {
    try {
      await api.deleteDocument(docId);
    } catch (err) {
      if (!(err instanceof ApiError && err.status === 404)) return setNotice(message(err));
    }
    const remaining = documents.filter((d) => d.doc_id !== docId);
    setDocuments(remaining);
    setThreads((all) => Object.fromEntries(Object.entries(all).filter(([k]) => !k.split(",").includes(docId))));
    setSelectedIds((ids) => {
      const next = ids.filter((id) => id !== docId);
      return next.length ? next : remaining[0] ? [remaining[0].doc_id] : [];
    });
    if (openSource?.doc_id === docId) setOpenSource(null);
  };

  const updateTurn = (k: string, turnId: string, patch: (t: Turn) => Partial<Turn>) =>
    setThreads((all) => ({ ...all, [k]: (all[k] ?? []).map((t) => (t.id === turnId ? { ...t, ...patch(t) } : t)) }));

  const runQuestion = async (k: string, docIds: string[], turnId: string, question: string) => {
    try {
      const answer = await api.askStream(docIds, question, {
        onSources: (sources) => updateTurn(k, turnId, () => ({ liveSources: sources })),
        onToken: (text) => updateTurn(k, turnId, (t) => ({ draft: (t.draft ?? "") + text })),
      });
      updateTurn(k, turnId, () => ({ state: "done", answer, draft: undefined, liveSources: undefined }));
    } catch (err) {
      updateTurn(k, turnId, () => ({ state: "error", error: message(err), draft: undefined, liveSources: undefined }));
    }
  };

  const ask = (question: string) => {
    if (!selectedDocs.length) return;
    const ids = selectedDocs.map((d) => d.doc_id);
    const turn: Turn = { id: crypto.randomUUID(), question, state: "pending" };
    setThreads((all) => ({ ...all, [key]: [...(all[key] ?? []), turn] }));
    void runQuestion(key, ids, turn.id, question);
  };

  const retry = (turn: Turn) => {
    if (!selectedDocs.length) return;
    updateTurn(key, turn.id, () => ({ state: "pending", error: undefined }));
    void runQuestion(key, selectedDocs.map((d) => d.doc_id), turn.id, turn.question);
  };

  const closeSource = useCallback(() => setOpenSource(null), []);

  const sidebar = (onClose?: () => void) => (
    <Sidebar
      documents={documents}
      loading={docsLoading}
      selectedIds={selectedIds}
      upload={upload}
      health={health}
      serverDown={serverDown}
      onSelect={selectOnly}
      onToggle={toggleSelected}
      onUpload={handleUpload}
      onDelete={handleDelete}
      onClose={onClose}
    />
  );

  return (
    <div className="flex h-dvh overflow-hidden">
      <div className="hidden w-72 shrink-0 lg:block">{sidebar()}</div>

      {drawerOpen && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Documents">
          <button type="button" aria-label="Close menu" className="absolute inset-0 bg-ink/25" onClick={() => setDrawerOpen(false)} />
          <div className="relative h-full w-[min(20rem,85vw)]">{sidebar(() => setDrawerOpen(false))}</div>
        </div>
      )}

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b border-line px-4 py-3 sm:px-6">
          <button type="button" onClick={() => setDrawerOpen(true)} className="rounded-md p-1.5 text-graphite hover:bg-line/60 lg:hidden" aria-label="Open documents">
            <MenuIcon />
          </button>
          <span className="lg:hidden">
            <LogoMark size={22} />
          </span>
          <div className="min-w-0">
            <h1 className="truncate font-semibold">
              {multiDoc ? `${selectedDocs.length} documents` : primaryDoc ? primaryDoc.title || primaryDoc.filename : "DocuMind AI"}
            </h1>
            {primaryDoc && (
              <p className="truncate text-xs text-graphite">
                {multiDoc
                  ? selectedDocs.map((d) => d.filename).join(", ")
                  : `${primaryDoc.page_count} pages, ${primaryDoc.chunk_count} indexed passages`}
              </p>
            )}
          </div>
        </header>

        {serverDown && <ServerDownBanner onRetry={connect} />}
        {notice && (
          <p role="alert" className="flex items-center justify-between gap-3 bg-danger-wash px-6 py-2.5 text-sm text-danger">
            {notice}
            <button type="button" className="font-medium underline" onClick={() => setNotice(null)}>
              Dismiss
            </button>
          </p>
        )}

        <div className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-3xl px-4 py-8 sm:px-6">
            {!primaryDoc ? (
              <Welcome
                upload={upload}
                onUpload={handleUpload}
                onSample={handleSample}
                sampleLoading={sampleLoading}
                disabled={serverDown}
                maxMb={maxUploadMb}
              />
            ) : turns.length === 0 ? (
              <DocumentIntro docs={selectedDocs} onAsk={ask} retrievalOnly={health?.llm_provider === "none"} />
            ) : (
              <div className="space-y-10">
                {turns.map((turn) => (
                  <Message
                    key={turn.id}
                    turn={turn}
                    multiDoc={multiDoc}
                    activeChunk={openSource ? `${openSource.doc_id}:${openSource.chunk_id}` : null}
                    onOpenSource={setOpenSource}
                    onRetry={() => retry(turn)}
                  />
                ))}
                <div ref={scrollEnd} />
              </div>
            )}
          </div>
        </div>

        <div className="border-t border-line bg-ivory px-4 pb-4 pt-3 sm:px-6">
          <div className="mx-auto max-w-3xl">
            <Composer
              disabled={!primaryDoc || serverDown}
              busy={asking}
              placeholder={
                multiDoc ? "Ask a question across these PDFs" : primaryDoc ? "Ask a question about this PDF" : "Upload a PDF to start asking questions"
              }
              onSubmit={ask}
            />
            <p className="mt-2 text-center text-xs text-mute">
              Answers come only from the selected {multiDoc ? "PDFs" : "PDF"}. Check the cited page before relying on them.
            </p>
          </div>
        </div>
      </main>

      {sourceDoc && openSource && (
        <div className="fixed inset-0 z-50 xl:static xl:z-auto xl:w-[440px] xl:shrink-0">
          <SourcePanel doc={sourceDoc} source={openSource} onClose={closeSource} />
        </div>
      )}
    </div>
  );
}

function ServerDownBanner({ onRetry }: { onRetry: () => void }) {
  return (
    <div role="alert" className="flex items-start gap-3 bg-danger-wash px-6 py-3 text-sm text-danger">
      <AlertIcon size={17} className="mt-0.5 shrink-0" />
      <p className="flex-1">
        Can&apos;t reach the DocuMind server. Start the backend (see the README), then reconnect.
      </p>
      <button type="button" onClick={onRetry} className="font-medium underline underline-offset-2">
        Reconnect
      </button>
    </div>
  );
}

function Welcome({
  upload,
  onUpload,
  onSample,
  sampleLoading,
  disabled,
  maxMb,
}: {
  upload: UploadState;
  onUpload: (f: File) => void;
  onSample: () => void;
  sampleLoading: boolean;
  disabled: boolean;
  maxMb: number;
}) {
  return (
    <section className="pt-6 sm:pt-12">
      <h2 className="max-w-[18ch] text-3xl font-semibold leading-[1.15] tracking-tight sm:text-[2.6rem]">
        Ask a PDF a question. See the page it came from.
      </h2>
      <p className="mt-4 max-w-[56ch] text-lg leading-relaxed text-graphite">
        DocuMind reads your document, finds the passages that answer you, and cites each one to its page. If the document
        doesn&apos;t say, it tells you so instead of guessing.
      </p>

      <div className="mt-9">
        <UploadDropzone state={upload} onFile={onUpload} maxMb={maxMb} />
      </div>

      <p className="mt-5 text-sm text-graphite">
        No PDF to hand?{" "}
        <button
          type="button"
          disabled={sampleLoading || disabled}
          onClick={onSample}
          className="font-medium text-blue-deep underline decoration-blue/40 underline-offset-[3px] hover:decoration-blue disabled:text-mute disabled:no-underline"
        >
          {sampleLoading ? "Loading the sample…" : "Try the sample library handbook"}
        </button>{" "}
        (a 7-page synthetic document).
      </p>

      <ExamplePreview />
    </section>
  );
}

/** A static illustration of what a cited answer looks like, labelled as an example. */
function ExamplePreview() {
  return (
    <figure className="mt-14 border-t border-line pt-8">
      <figcaption className="mb-4 text-sm text-graphite">What an answer looks like</figcaption>
      <div className="grid gap-5 sm:grid-cols-[1fr_15rem]">
        <p className="leading-[1.7]">
          Overdue books don&apos;t collect daily fines; the library has been fine-free since March 2023.
          <span className="mx-1 inline-flex rounded-md bg-wash px-1.5 text-[0.78rem] font-semibold text-blue-deep">p.&nbsp;4</span>
          After 45 days an item is marked lost and billed at replacement cost.
          <span className="mx-1 inline-flex rounded-md bg-wash px-1.5 text-[0.78rem] font-semibold text-blue-deep">p.&nbsp;4</span>
        </p>
        <blockquote className="doc-quote rounded-lg border border-line bg-paper p-4 text-sm text-graphite">
          <span className="rounded-sm bg-wash text-ink">Harbor City Public Library has been fine-free since March 2023</span>: there are no daily fines for
          overdue items.
        </blockquote>
      </div>
    </figure>
  );
}

function DocumentIntro({ docs, onAsk, retrievalOnly }: { docs: DocumentInfo[]; onAsk: (q: string) => void; retrievalOnly: boolean }) {
  const single = docs.length === 1 ? docs[0] : null;
  const isSample = single?.filename === SAMPLE_FILENAME;
  const scanned = docs.filter((d) => d.image_only_pages.length > 0);
  return (
    <section className="pt-6 sm:pt-10">
      <h2 className="text-2xl font-semibold tracking-tight">What do you want to know?</h2>
      {single ? (
        <p className="mt-2 max-w-[60ch] text-graphite">
          Ask about anything in <span className="font-medium text-ink">{single.filename}</span>. Each answer links to the pages it
          used; select a page number to see the passage highlighted. To search several PDFs at once, tick them in the sidebar.
        </p>
      ) : (
        <>
          <p className="mt-2 max-w-[60ch] text-graphite">
            Questions are answered from all {docs.length} selected documents. Each citation names the document and page it came from.
          </p>
          <ul className="mt-4 space-y-1 text-sm">
            {docs.map((d) => (
              <li key={d.doc_id} className="text-ink">
                {d.filename} <span className="text-graphite">({d.page_count} pages)</span>
              </li>
            ))}
          </ul>
        </>
      )}
      {scanned.map((d) => (
        <p key={d.doc_id} className="mt-4 rounded-xl bg-caution-wash px-4 py-3 text-sm text-caution">
          {docs.length > 1 ? `${d.filename}: ` : ""}
          {d.image_only_pages.length === 1 ? "Page" : "Pages"} {d.image_only_pages.join(", ")} look scanned and have no text layer, so
          they can&apos;t be searched.
        </p>
      ))}
      {isSample && (
        <div className="mt-8">
          <h3 className="mb-3 text-sm font-medium text-graphite">Try one of these</h3>
          <ul className="flex flex-col items-start gap-2">
            {SAMPLE_QUESTIONS.map((q) => (
              <li key={q}>
                <button
                  type="button"
                  onClick={() => onAsk(q)}
                  className="rounded-full border border-line-strong bg-paper px-4 py-2 text-left text-sm transition-colors hover:border-blue hover:text-blue-deep"
                >
                  {q}
                </button>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-xs text-mute">
            {retrievalOnly
              ? "The last one isn't answered in the handbook. With a language model configured, DocuMind declines it."
              : "The last one isn't answered in the handbook, so DocuMind should decline it."}
          </p>
        </div>
      )}
    </section>
  );
}
