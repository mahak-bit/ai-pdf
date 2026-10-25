// Issues short-lived, narrowly scoped permission for the browser to upload one PDF
// straight to Vercel Blob. This bypasses the 4.5 MB request-body cap of Vercel
// Functions; the API then ingests the file from Blob (POST /api/documents/from-blob).
// Authenticates to Blob with the project's Vercel OIDC token and BLOB_STORE_ID.
import { issueSignedToken } from "@vercel/blob";
import { handleUploadPresigned, type HandleUploadPresignedBody } from "@vercel/blob/client";

const MAX_BYTES = 20 * 1024 * 1024;
const UPLOAD_PATH = /^uploads\/[0-9a-f-]{36}\.pdf$/;
const TOKEN_LIFETIME_MS = 10 * 60 * 1000;

export async function POST(request: Request): Promise<Response> {
  let body: HandleUploadPresignedBody;
  try {
    body = (await request.json()) as HandleUploadPresignedBody;
  } catch {
    return Response.json({ error: { code: "bad_request", message: "Invalid upload request." } }, { status: 400 });
  }

  try {
    const result = await handleUploadPresigned({
      body,
      request,
      getSignedToken: async (pathname) => {
        if (!UPLOAD_PATH.test(pathname)) throw new Error("Uploads must go to uploads/<uuid>.pdf");
        const validUntil = Date.now() + TOKEN_LIFETIME_MS;
        const options = { allowedContentTypes: ["application/pdf"], maximumSizeInBytes: MAX_BYTES, validUntil };
        const token = await issueSignedToken({ pathname, operations: ["put"], ...options });
        return { token, urlOptions: options };
      },
    });
    return Response.json(result);
  } catch (error) {
    console.error("Upload token request failed", error);
    return Response.json(
      { error: { code: "upload_not_allowed", message: "The upload could not be started. Try again." } },
      { status: 400 },
    );
  }
}
