// Typed client for the DocuMind FastAPI backend. The browser talks to the backend
// directly; no secrets live here (the Gemini key stays on the server).

export const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");

export type DocumentInfo = {
  doc_id: string;
  filename: string;
  page_count: number;
  chunk_count: number;
  char_count: number;
  created_at: string;
  title: string | null;
  image_only_pages: number[];
};

export type Source = {
  source_id: string;
  chunk_id: string;
  page: number;
  excerpt: string;
  score: number;
};

export type AnswerStatus = "answered" | "abstained" | "retrieval_only";

export type Answer = {
  status: AnswerStatus;
  answer: string;
  sources: Source[];
  model: string | null;
  retrieved: number;
};

export type Health = {
  status: "ok";
  llm_provider: "gemini" | "ollama" | "none" | string;
  llm_model: string | null;
  embedding_model: string;
  embedding_model_loaded: boolean;
  documents: number;
  max_upload_mb: number;
  storage: "local" | "blob";
};

export class ApiError extends Error {
  constructor(
    message: string,
    public code: string,
    public status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const NETWORK_MESSAGE = `Can't reach the DocuMind server at ${API_BASE_URL}. Check that the backend is running.`;

export async function toApiError(response: Response): Promise<ApiError> {
  try {
    const body = await response.json();
    if (body?.error?.message) return new ApiError(body.error.message, body.error.code ?? "error", response.status);
  } catch {
    // Non-JSON body (proxy error page, etc.)
  }
  return new ApiError(`The server returned an error (HTTP ${response.status}).`, "http_error", response.status);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(NETWORK_MESSAGE, "network_error", 0);
  }
  if (!response.ok) throw await toApiError(response);
  return (response.status === 204 ? undefined : await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/api/health"),
  listDocuments: () => request<DocumentInfo[]>("/api/documents"),
  loadSample: () => request<DocumentInfo>("/api/documents/sample", { method: "POST" }),
  deleteDocument: (docId: string) => request<void>(`/api/documents/${docId}`, { method: "DELETE" }),
  ask: (docId: string, question: string, signal?: AbortSignal) =>
    request<Answer>(`/api/documents/${docId}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal,
    }),
  pageImageUrl: (docId: string, page: number, chunkId?: string) =>
    `${API_BASE_URL}/api/documents/${docId}/pages/${page}${chunkId ? `?chunk_id=${encodeURIComponent(chunkId)}` : ""}`,

  /** Upload with progress. XHR is used because fetch has no upload progress events. */
  uploadDocument(file: File, onProgress?: (fraction: number) => void): Promise<DocumentInfo> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE_URL}/api/documents`);
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) onProgress?.(event.loaded / event.total);
      };
      xhr.onerror = () => reject(new ApiError(NETWORK_MESSAGE, "network_error", 0));
      xhr.onload = () => {
        let body: unknown = null;
        try {
          body = JSON.parse(xhr.responseText);
        } catch {
          // fall through to the generic error below
        }
        if (xhr.status >= 200 && xhr.status < 300 && body) return resolve(body as DocumentInfo);
        const err = (body as { error?: { message?: string; code?: string } } | null)?.error;
        // Hosting platforms may reject large bodies before the API sees them (plain-text 413).
        if (!err && xhr.status === 413)
          return reject(new ApiError("That file is larger than the server's upload limit.", "file_too_large", 413));
        reject(
          new ApiError(
            err?.message ?? `Upload failed (HTTP ${xhr.status}).`,
            err?.code ?? "http_error",
            xhr.status,
          ),
        );
      };
      const form = new FormData();
      form.append("file", file);
      xhr.send(form);
    });
  },
};

/** Used until /api/health reports the server's limit (4 MB on Vercel, 20 MB elsewhere). */
export const DEFAULT_MAX_UPLOAD_MB = 20;

/** Client-side checks mirror the server's, so obvious mistakes fail instantly. */
export function checkFile(file: File, maxMb: number = DEFAULT_MAX_UPLOAD_MB): string | null {
  if (!file.name.toLowerCase().endsWith(".pdf")) return "Only PDF files are supported. Choose a file ending in .pdf.";
  if (file.size === 0) return "That file is empty.";
  if (file.size > maxMb * 1024 * 1024) return `That file is larger than the ${maxMb} MB limit.`;
  return null;
}
