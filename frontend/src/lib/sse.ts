export type SSEEvent = { event: string; data: unknown };

/**
 * Parse complete server-sent events out of a buffer. Network chunks can split an event
 * anywhere, so anything after the last blank line is returned as `rest` to prepend to
 * the next chunk.
 */
export function parseSSE(buffer: string): { events: SSEEvent[]; rest: string } {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const blocks = normalized.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: SSEEvent[] = [];
  for (const block of blocks) {
    let event = "message";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    if (!data.length) continue;
    try {
      events.push({ event, data: JSON.parse(data.join("\n")) });
    } catch {
      // Ignore malformed events rather than breaking the whole stream.
    }
  }
  return { events, rest };
}
