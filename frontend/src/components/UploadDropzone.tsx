"use client";

import { useRef, useState } from "react";
import { DEFAULT_MAX_UPLOAD_MB } from "@/lib/api";
import { UploadIcon } from "./icons";

export type UploadState =
  | { phase: "idle"; error?: string }
  | { phase: "uploading"; filename: string; progress: number }
  | { phase: "indexing"; filename: string };

type Props = {
  state: UploadState;
  onFile: (file: File) => void;
  compact?: boolean;
  maxMb?: number;
};

export function UploadDropzone({ state, onFile, compact = false, maxMb = DEFAULT_MAX_UPLOAD_MB }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const busy = state.phase !== "idle";

  const pick = (files: FileList | null) => {
    const file = files?.[0];
    if (file) onFile(file);
    if (input.current) input.current.value = "";
  };

  return (
    <div>
      <button
        type="button"
        disabled={busy}
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          if (!busy) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!busy) pick(e.dataTransfer.files);
        }}
        className={[
          "group w-full rounded-xl border border-dashed text-left transition-colors",
          compact ? "px-3.5 py-3" : "px-6 py-9 sm:px-10",
          dragging ? "border-blue bg-wash" : "border-line-strong bg-paper hover:border-blue/60",
          busy ? "cursor-progress" : "cursor-pointer",
        ].join(" ")}
      >
        {state.phase === "idle" ? (
          <span className={`flex items-center ${compact ? "gap-2.5" : "flex-col gap-3 text-center"}`}>
            <span
              className={`grid place-items-center rounded-full text-blue ${compact ? "size-7 bg-wash" : "size-11 bg-wash"}`}
            >
              <UploadIcon size={compact ? 15 : 20} />
            </span>
            <span>
              <span className={`block font-medium ${compact ? "text-sm" : "text-lg"}`}>
                {compact ? "Upload PDF" : "Drop a PDF here, or choose a file"}
              </span>
              {!compact && (
                <span className="mt-1 block text-sm text-graphite">
                  Text-based PDFs up to {maxMb} MB. Scanned pages need OCR first.
                </span>
              )}
            </span>
          </span>
        ) : (
          <UploadProgress state={state} compact={compact} />
        )}
      </button>
      <input
        ref={input}
        type="file"
        accept="application/pdf,.pdf"
        className="sr-only"
        tabIndex={-1}
        onChange={(e) => pick(e.target.files)}
      />
      {state.phase === "idle" && state.error && (
        <p role="alert" className="mt-2 text-sm text-danger">
          {state.error}
        </p>
      )}
    </div>
  );
}

function UploadProgress({ state, compact }: { state: Exclude<UploadState, { phase: "idle" }>; compact: boolean }) {
  const uploading = state.phase === "uploading";
  const percent = uploading ? Math.round(state.progress * 100) : 100;
  return (
    <span className={`block ${compact ? "" : "mx-auto max-w-sm text-center"}`} aria-live="polite">
      <span className="block truncate text-sm font-medium">{state.filename}</span>
      <span className="mt-1 block text-sm text-graphite">
        {uploading ? `Uploading… ${percent}%` : "Reading pages and building the search index…"}
      </span>
      <span className="mt-2.5 block h-1 overflow-hidden rounded-full bg-line">
        <span
          className={`block h-full rounded-full bg-blue transition-[width] duration-300 ${uploading ? "" : "animate-pulse"}`}
          style={{ width: `${percent}%` }}
        />
      </span>
    </span>
  );
}
