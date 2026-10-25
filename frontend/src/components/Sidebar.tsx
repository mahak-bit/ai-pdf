"use client";

import { useState } from "react";
import type { DocumentInfo, Health } from "@/lib/api";
import { CloseIcon, LogoMark, TrashIcon } from "./icons";
import { UploadDropzone, type UploadState } from "./UploadDropzone";

type Props = {
  documents: DocumentInfo[];
  loading: boolean;
  selectedIds: string[];
  upload: UploadState;
  health: Health | null;
  serverDown: boolean;
  /** Make this the only selected document. */
  onSelect: (docId: string) => void;
  /** Add or remove a document from a multi-document selection. */
  onToggle: (docId: string) => void;
  onUpload: (file: File) => void;
  onDelete: (docId: string) => void;
  onClose?: () => void;
};

export function Sidebar({ documents, loading, selectedIds, upload, health, serverDown, onSelect, onToggle, onUpload, onDelete, onClose }: Props) {
  const [confirming, setConfirming] = useState<string | null>(null);

  return (
    <nav aria-label="Documents" className="flex h-full flex-col border-r border-line bg-ivory">
      <div className="flex items-center justify-between px-5 pb-5 pt-5">
        <div className="flex items-center gap-2.5">
          <LogoMark />
          <span className="text-lg font-semibold tracking-tight">DocuMind</span>
        </div>
        {onClose && (
          <button type="button" onClick={onClose} className="rounded-md p-1.5 text-graphite hover:bg-line/60 lg:hidden" aria-label="Close menu">
            <CloseIcon />
          </button>
        )}
      </div>

      <div className="px-4">
        <UploadDropzone compact state={upload} onFile={onUpload} />
      </div>

      <h2 className="px-5 pt-6 text-sm font-medium text-graphite">Your documents</h2>
      <p className="px-5 pb-2 text-xs text-mute">{documents.length > 1 ? "Tick several to ask across them." : " "}</p>
      <ul className="min-h-0 flex-1 space-y-0.5 overflow-y-auto px-2.5 pb-4">
        {loading && <li className="px-2.5 py-2 text-sm text-mute">Loading…</li>}
        {!loading && documents.length === 0 && (
          <li className="px-2.5 py-2 text-sm text-mute">Nothing uploaded yet.</li>
        )}
        {documents.map((doc) => {
          const active = selectedIds.includes(doc.doc_id);
          return (
            <li key={doc.doc_id} className="group relative">
              <input
                type="checkbox"
                checked={active}
                onChange={() => onToggle(doc.doc_id)}
                aria-label={`Include ${doc.filename} in questions`}
                className="absolute left-3 top-3 z-10 size-4 cursor-pointer accent-[var(--color-blue)]"
              />
              <button
                type="button"
                onClick={() => onSelect(doc.doc_id)}
                aria-current={active ? "true" : undefined}
                className={[
                  "flex w-full items-start gap-2.5 rounded-lg py-2 pl-9 pr-9 text-left transition-colors",
                  active ? "bg-paper shadow-[inset_0_0_0_1px_var(--color-line)]" : "hover:bg-line/40",
                ].join(" ")}
              >
                <span className="min-w-0">
                  <span className="block truncate text-sm font-medium">{doc.filename}</span>
                  <span className="block text-xs text-graphite">
                    {doc.page_count} {doc.page_count === 1 ? "page" : "pages"}
                  </span>
                </span>
              </button>
              {confirming === doc.doc_id ? (
                <span className="absolute right-1.5 top-1.5 flex gap-1 rounded-md bg-paper p-0.5 shadow-[0_0_0_1px_var(--color-line)]">
                  <button
                    type="button"
                    className="rounded px-2 py-1 text-xs font-medium text-danger hover:bg-danger-wash"
                    onClick={() => {
                      setConfirming(null);
                      onDelete(doc.doc_id);
                    }}
                  >
                    Delete
                  </button>
                  <button type="button" className="rounded px-2 py-1 text-xs text-graphite hover:bg-line/50" onClick={() => setConfirming(null)}>
                    Keep
                  </button>
                </span>
              ) : (
                <button
                  type="button"
                  onClick={() => setConfirming(doc.doc_id)}
                  aria-label={`Delete ${doc.filename}`}
                  className="absolute right-1.5 top-2 rounded-md p-1.5 text-mute opacity-100 hover:bg-line/60 hover:text-danger focus-visible:opacity-100 lg:opacity-0 lg:group-hover:opacity-100"
                >
                  <TrashIcon size={15} />
                </button>
              )}
            </li>
          );
        })}
      </ul>

      <ModelStatus health={health} serverDown={serverDown} />
    </nav>
  );
}

function ModelStatus({ health, serverDown }: { health: Health | null; serverDown: boolean }) {
  let dot = "bg-mute";
  let text = "Connecting to server…";
  if (serverDown) {
    dot = "bg-danger";
    text = "Server unreachable";
  } else if (health) {
    if (health.llm_provider === "none") {
      dot = "bg-[#c99a2e]";
      text = "Retrieval only: no language model configured";
    } else {
      dot = "bg-[#2f9e5b]";
      text = `Answers by ${health.llm_model ?? health.llm_provider}`;
    }
  }
  return (
    <p className="flex items-center gap-2 border-t border-line px-5 py-3.5 text-xs text-graphite" aria-live="polite">
      <span className={`size-2 shrink-0 rounded-full ${dot}`} aria-hidden="true" />
      {text}
    </p>
  );
}
