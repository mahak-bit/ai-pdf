import { describe, expect, it } from "vitest";
import { parseSSE } from "./sse";

describe("parseSSE", () => {
  it("parses complete events and keeps a partial one for the next chunk", () => {
    const { events, rest } = parseSSE('event: token\ndata: {"text":"Hel"}\n\nevent: token\ndata: {"te');
    expect(events).toEqual([{ event: "token", data: { text: "Hel" } }]);
    expect(rest).toBe('event: token\ndata: {"te');
  });

  it("reassembles an event split across chunks", () => {
    const first = parseSSE('event: final\ndata: {"status":');
    const second = parseSSE(first.rest + '"answered"}\n\n');
    expect(first.events).toEqual([]);
    expect(second.events).toEqual([{ event: "final", data: { status: "answered" } }]);
    expect(second.rest).toBe("");
  });

  it("handles CRLF line endings and skips malformed data", () => {
    const { events } = parseSSE('event: token\r\ndata: {"text":"a"}\r\n\r\nevent: token\r\ndata: not-json\r\n\r\n');
    expect(events).toEqual([{ event: "token", data: { text: "a" } }]);
  });
});
