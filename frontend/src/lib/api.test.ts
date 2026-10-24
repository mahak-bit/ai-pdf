import { describe, expect, it } from "vitest";
import { checkFile, toApiError } from "./api";

const file = (name: string, size: number) => new File([new Uint8Array(size)], name, { type: "application/pdf" });

describe("checkFile", () => {
  it("accepts a normal PDF", () => {
    expect(checkFile(file("report.PDF", 1024))).toBeNull();
  });

  it("rejects other extensions, empty and oversized files", () => {
    expect(checkFile(file("notes.docx", 10))).toMatch(/Only PDF/);
    expect(checkFile(file("empty.pdf", 0))).toMatch(/empty/);
    expect(checkFile(file("huge.pdf", 21 * 1024 * 1024))).toMatch(/20 MB/);
  });

  it("uses the server's limit when given", () => {
    expect(checkFile(file("doc.pdf", 5 * 1024 * 1024), 4)).toMatch(/4 MB/);
    expect(checkFile(file("doc.pdf", 3 * 1024 * 1024), 4)).toBeNull();
  });
});

describe("toApiError", () => {
  it("uses the server's error envelope", async () => {
    const res = new Response(JSON.stringify({ error: { code: "scanned_pdf", message: "This PDF looks scanned." } }), { status: 422 });
    const err = await toApiError(res);
    expect(err).toMatchObject({ code: "scanned_pdf", message: "This PDF looks scanned.", status: 422 });
  });

  it("falls back for non-JSON bodies", async () => {
    const err = await toApiError(new Response("<html>Bad gateway</html>", { status: 502 }));
    expect(err.message).toBe("The server returned an error (HTTP 502).");
  });
});
