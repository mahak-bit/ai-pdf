import type { Source } from "./api";

export type Segment = { kind: "text"; text: string } | { kind: "cite"; source: Source };

const MARKER = /\[(S\d+)\]/g;

/**
 * Split an answer into text and citation segments. Markers that don't match a returned
 * source are dropped (the server already validates them; this is a second guard so a
 * citation chip can never point at nothing). Adjacent duplicates collapse to one chip.
 */
export function segmentAnswer(answer: string, sources: Source[]): Segment[] {
  const byId = new Map(sources.map((s) => [s.source_id, s]));
  const segments: Segment[] = [];
  let last = 0;
  for (const match of answer.matchAll(MARKER)) {
    const index = match.index ?? 0;
    if (index > last) segments.push({ kind: "text", text: answer.slice(last, index) });
    last = index + match[0].length;
    const source = byId.get(match[1]);
    if (!source) continue;
    const prev = segments[segments.length - 1];
    if (prev?.kind === "cite" && prev.source.source_id === source.source_id) continue;
    segments.push({ kind: "cite", source });
  }
  if (last < answer.length) segments.push({ kind: "text", text: answer.slice(last) });
  return mergeText(segments);
}

function mergeText(segments: Segment[]): Segment[] {
  const out: Segment[] = [];
  for (const seg of segments) {
    const prev = out[out.length - 1];
    if (seg.kind === "text" && prev?.kind === "text") prev.text += seg.text;
    else out.push(seg.kind === "text" ? { ...seg } : seg);
  }
  return out;
}

/** "p. 4" or "pp. 2, 5" for a set of sources. */
export function pageLabel(pages: number[]): string {
  const unique = [...new Set(pages)].sort((a, b) => a - b);
  return unique.length === 1 ? `p. ${unique[0]}` : `pp. ${unique.join(", ")}`;
}

/** Trim an excerpt to roughly `max` characters on a word boundary. */
export function shorten(text: string, max = 220): string {
  if (text.length <= max) return text;
  const cut = text.slice(0, max);
  return `${cut.slice(0, cut.lastIndexOf(" ")).trimEnd()}…`;
}
