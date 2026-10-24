import { describe, expect, it } from "vitest";
import type { Source } from "./api";
import { pageLabel, segmentAnswer, shorten } from "./citations";

const src = (id: string, page: number): Source => ({ source_id: id, chunk_id: `p${page}-c0`, page, excerpt: "x", score: 0.5 });

describe("segmentAnswer", () => {
  it("turns markers into citation segments tied to their source", () => {
    const segs = segmentAnswer("Fees are $35 [S1]. Hours vary [S2].", [src("S1", 2), src("S2", 6)]);
    expect(segs).toEqual([
      { kind: "text", text: "Fees are $35 " },
      { kind: "cite", source: src("S1", 2) },
      { kind: "text", text: ". Hours vary " },
      { kind: "cite", source: src("S2", 6) },
      { kind: "text", text: "." },
    ]);
  });

  it("drops markers that do not match a returned source", () => {
    const segs = segmentAnswer("Claim [S9] here.", [src("S1", 1)]);
    expect(segs).toEqual([{ kind: "text", text: "Claim  here." }]);
  });

  it("collapses repeated adjacent citations", () => {
    const segs = segmentAnswer("A [S1][S1].", [src("S1", 3)]);
    expect(segs.filter((s) => s.kind === "cite")).toHaveLength(1);
  });

  it("returns plain text when there are no markers", () => {
    expect(segmentAnswer("No citations.", [])).toEqual([{ kind: "text", text: "No citations." }]);
  });
});

describe("helpers", () => {
  it("formats page labels", () => {
    expect(pageLabel([4])).toBe("p. 4");
    expect(pageLabel([5, 2, 5])).toBe("pp. 2, 5");
  });

  it("shortens on a word boundary", () => {
    expect(shorten("short")).toBe("short");
    expect(shorten("one two three four five", 12)).toBe("one two…");
  });
});
