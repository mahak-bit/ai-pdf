"use client";

import { useEffect, useRef, useState } from "react";
import { api, type DocumentInfo, type Source } from "@/lib/api";
import { CloseIcon } from "./icons";

type Props = {
  doc: DocumentInfo;
  source: Source;
  onClose: () => void;
};

/** The cited page, rendered by the backend with the passage highlighted, plus the exact excerpt. */
export function SourcePanel({ doc, source, onClose }: Props) {
  const closeButton = useRef<HTMLButtonElement>(null);
  const src = api.pageImageUrl(doc.doc_id, source.page, source.chunk_id);

  useEffect(() => {
    closeButton.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [source.chunk_id, onClose]);

  return (
    <aside aria-label={`Source on page ${source.page}`} className="panel-in flex h-full flex-col border-l border-line bg-paper">
      <header className="flex items-center justify-between border-b border-line px-5 py-3.5">
        <div>
          <h2 className="font-semibold">
            Page {source.page} <span className="font-normal text-graphite">of {doc.page_count}</span>
          </h2>
          <p className="truncate text-xs text-graphite">{doc.filename}</p>
        </div>
        <button ref={closeButton} type="button" onClick={onClose} aria-label="Close source" className="rounded-md p-1.5 text-graphite hover:bg-ivory">
          <CloseIcon />
        </button>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto">
        <section className="border-b border-line px-5 py-5">
          <h3 className="mb-2 text-sm font-medium text-graphite">Cited passage</h3>
          <blockquote className="doc-quote border-l-2 border-blue pl-4">{source.excerpt}</blockquote>
          <p className="mt-3 text-xs text-mute">Similarity to your question: {source.score.toFixed(2)}</p>
        </section>
        <section className="bg-ivory px-5 py-5">
          <h3 className="mb-3 text-sm font-medium text-graphite">On the page</h3>
          {/* key resets loading state when switching citations */}
          <PageImage key={src} src={src} page={source.page} />
        </section>
      </div>
    </aside>
  );
}

function PageImage({ src, page }: { src: string; page: number }) {
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  return (
    <div className="relative overflow-hidden rounded-lg border border-line bg-paper">
      {status === "loading" && <div className="aspect-[8.5/11] w-full animate-pulse bg-line/40" aria-label="Loading page" />}
      {status === "error" && (
        <p className="aspect-[8.5/11] w-full p-6 text-sm text-graphite">The page image couldn&apos;t be loaded. The cited passage is shown above.</p>
      )}
      {/* eslint-disable-next-line @next/next/no-img-element -- dynamic PNG from the API; next/image adds nothing here */}
      <img
        src={src}
        alt={`Page ${page} of the document with the cited passage highlighted`}
        onLoad={() => setStatus("ready")}
        onError={() => setStatus("error")}
        className={status === "ready" ? "block w-full" : "hidden"}
      />
    </div>
  );
}
