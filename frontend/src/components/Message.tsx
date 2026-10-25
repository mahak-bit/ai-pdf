"use client";

import type { Answer, Source } from "@/lib/api";
import { segmentAnswer, shorten } from "@/lib/citations";
import { AlertIcon, InfoIcon } from "./icons";

export type Turn = {
  id: string;
  question: string;
  state: "pending" | "done" | "error";
  answer?: Answer;
  error?: string;
  /** Answer text streamed so far; replaced by the validated `answer` when it arrives. */
  draft?: string;
  /** Passages retrieved for this question, sent before generation starts. */
  liveSources?: Source[];
};

type Props = {
  turn: Turn;
  multiDoc: boolean;
  /** `${doc_id}:${chunk_id}` of the source open in the side panel. */
  activeChunk: string | null;
  onOpenSource: (source: Source) => void;
  onRetry: () => void;
};

const sourceKey = (s: Source) => `${s.doc_id}:${s.chunk_id}`;

/** "annual-report…" from "annual-report-2026-final.pdf", for compact citation chips. */
function shortName(filename: string, max = 18): string {
  const base = filename.replace(/\.pdf$/i, "");
  return base.length > max ? `${base.slice(0, max - 1)}…` : base;
}

export function Message({ turn, multiDoc, activeChunk, onOpenSource, onRetry }: Props) {
  return (
    <article className="space-y-4">
      <div className="flex justify-end">
        <p className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md border border-line bg-paper px-4 py-2.5">{turn.question}</p>
      </div>
      <div className="max-w-[68ch]">
        {turn.state === "pending" &&
          (turn.draft ? (
            <AnswerText
              text={turn.draft}
              sources={turn.liveSources ?? []}
              multiDoc={multiDoc}
              activeChunk={activeChunk}
              onOpenSource={onOpenSource}
              streaming
            />
          ) : (
            <Thinking label={turn.liveSources ? "Writing the answer…" : "Searching the documents…"} />
          ))}
        {turn.state === "error" && (
          <div role="alert" className="flex gap-3 rounded-xl bg-danger-wash px-4 py-3 text-danger">
            <AlertIcon className="mt-0.5 shrink-0" />
            <div>
              <p>{turn.error}</p>
              <button type="button" onClick={onRetry} className="mt-1.5 text-sm font-medium underline underline-offset-2">
                Try again
              </button>
            </div>
          </div>
        )}
        {turn.state === "done" && turn.answer && (
          <AnswerBody answer={turn.answer} multiDoc={multiDoc} activeChunk={activeChunk} onOpenSource={onOpenSource} />
        )}
      </div>
    </article>
  );
}

function Thinking({ label }: { label: string }) {
  return (
    <p className="flex items-center gap-2 text-graphite" role="status">
      <span className="flex gap-1" aria-hidden="true">
        {[0, 1, 2].map((i) => (
          <span key={i} className="thinking-dot size-1.5 rounded-full bg-blue" style={{ animationDelay: `${i * 0.15}s` }} />
        ))}
      </span>
      {label}
    </p>
  );
}

type AnswerViewProps = {
  multiDoc: boolean;
  activeChunk: string | null;
  onOpenSource: (s: Source) => void;
};

function AnswerText({ text, sources, streaming = false, ...view }: AnswerViewProps & { text: string; sources: Source[]; streaming?: boolean }) {
  // While streaming, hide a citation marker that has only partly arrived ("[S", "[S1").
  const segments = segmentAnswer(streaming ? text.replace(/\[S?\d*$/, "") : text, sources);
  return (
    <p className="whitespace-pre-wrap text-[1.0rem] leading-[1.7]" aria-live={streaming ? "polite" : undefined} aria-busy={streaming}>
      {segments.map((seg, i) =>
        seg.kind === "text" ? (
          <span key={i}>{seg.text}</span>
        ) : (
          <CitationChip
            key={i}
            source={seg.source}
            multiDoc={view.multiDoc}
            active={sourceKey(seg.source) === view.activeChunk}
            onClick={() => view.onOpenSource(seg.source)}
          />
        ),
      )}
      {streaming && <span className="stream-caret ml-0.5 inline-block h-[1.05em] w-[2px] translate-y-[3px] bg-blue" aria-hidden="true" />}
    </p>
  );
}

function AnswerBody({ answer, ...view }: AnswerViewProps & { answer: Answer }) {
  if (answer.status === "abstained") {
    return (
      <div className="flex gap-3 rounded-xl border border-line bg-paper px-4 py-3">
        <InfoIcon className="mt-0.5 shrink-0 text-graphite" />
        <div>
          <p>{answer.answer}</p>
          <p className="mt-1 text-sm text-graphite">DocuMind only answers from what the documents say.</p>
        </div>
      </div>
    );
  }

  if (answer.status === "retrieval_only") {
    return (
      <div className="space-y-3">
        <p className="rounded-xl bg-caution-wash px-4 py-3 text-sm text-caution">{answer.answer}</p>
        <SourceList sources={answer.sources} {...view} />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <AnswerText text={answer.answer} sources={answer.sources} {...view} />
      <SourceList sources={answer.sources} {...view} />
      {answer.model && (
        <p className="text-xs text-mute">
          Generated by {answer.model.split(":").slice(1).join(":")} from {answer.retrieved} retrieved passages
        </p>
      )}
    </div>
  );
}

function CitationChip({ source, multiDoc, active, onClick }: { source: Source; multiDoc: boolean; active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={`Show source: ${source.filename}, page ${source.page}`}
      aria-pressed={active}
      className={[
        "mx-0.5 inline-flex translate-y-[-1px] items-center rounded-md px-1.5 py-px align-baseline text-[0.78rem] font-semibold tabular-nums transition-colors",
        active ? "bg-blue text-white" : "bg-wash text-blue-deep hover:bg-blue hover:text-white",
      ].join(" ")}
    >
      {multiDoc && <span className="mr-1 max-w-[12ch] truncate font-medium">{shortName(source.filename)}</span>}
      p.&nbsp;{source.page}
    </button>
  );
}

function SourceList({ sources, multiDoc, activeChunk, onOpenSource }: AnswerViewProps & { sources: Source[] }) {
  if (!sources.length) return null;
  return (
    <div>
      <h3 className="mb-2 text-sm font-medium text-graphite">{sources.length === 1 ? "Source" : `${sources.length} sources`}</h3>
      <ul className="divide-y divide-line overflow-hidden rounded-xl border border-line bg-paper">
        {sources.map((s) => {
          const active = sourceKey(s) === activeChunk;
          return (
            <li key={sourceKey(s)}>
              <button
                type="button"
                onClick={() => onOpenSource(s)}
                className={`flex w-full gap-4 px-4 py-3 text-left transition-colors ${active ? "bg-wash" : "hover:bg-ivory"}`}
              >
                <span className="w-12 shrink-0 pt-0.5 text-sm font-semibold tabular-nums text-blue-deep">p.&nbsp;{s.page}</span>
                <span className="min-w-0">
                  {multiDoc && <span className="mb-0.5 block truncate text-xs font-medium text-graphite">{s.filename}</span>}
                  <span className="doc-quote block text-graphite">{shorten(s.excerpt, 200)}</span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
