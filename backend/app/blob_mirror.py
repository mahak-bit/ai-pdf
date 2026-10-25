"""Optional mirror of document folders to Vercel Blob.

Serverless instances don't share a disk, so on Vercel each document's files
(PDF, chunks, index, metadata) are also stored in a private Blob store under
``documents/<doc_id>/``. Any instance can then download a document it hasn't
seen. Locally (no BLOB_READ_WRITE_TOKEN) the mirror is disabled and the disk is
the only store.
"""

import logging
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

PREFIX = "documents/"


class DocumentMirror(Protocol):
    def upload(self, doc_id: str, files: dict[str, bytes]) -> None: ...

    def download(self, doc_id: str, dest: Path) -> bool:
        """Write the document's files into ``dest``. Returns False if it doesn't exist."""
        ...

    def read_meta_files(self) -> dict[str, bytes]:
        """Return {doc_id: meta.json bytes} for every mirrored document."""
        ...

    def delete(self, doc_id: str) -> None: ...

    def read_upload(self, pathname: str) -> bytes:
        """Bytes of a file the browser uploaded directly. Raises FileNotFoundError."""
        ...

    def delete_upload(self, pathname: str) -> None: ...


class VercelBlobMirror:
    def __init__(self, access: str = "private"):
        from vercel.blob import sync as blob  # imported lazily: only needed on Vercel

        self._blob = blob
        self.access = access

    def upload(self, doc_id: str, files: dict[str, bytes]) -> None:
        for name, body in files.items():
            self._blob.put_blob(
                f"{PREFIX}{doc_id}/{name}", body, access=self.access, allow_overwrite=True,
                content_type="application/pdf" if name.endswith(".pdf") else "application/octet-stream",
            )

    def download(self, doc_id: str, dest: Path) -> bool:
        items = list(self._blob.query_blobs(prefix=f"{PREFIX}{doc_id}/"))
        if not items:
            return False
        dest.mkdir(parents=True, exist_ok=True)
        for item in items:
            name = item.pathname.rsplit("/", 1)[-1]
            (dest / name).write_bytes(self._blob.get_blob(item.pathname, access=self.access).body)
        return True

    def read_meta_files(self) -> dict[str, bytes]:
        metas: dict[str, bytes] = {}
        for item in self._blob.query_blobs(prefix=PREFIX):
            if item.pathname.endswith("/meta.json"):
                doc_id = item.pathname[len(PREFIX):].split("/", 1)[0]
                metas[doc_id] = self._blob.get_blob(item.pathname, access=self.access).body
        return metas

    def delete(self, doc_id: str) -> None:
        paths = [item.pathname for item in self._blob.query_blobs(prefix=f"{PREFIX}{doc_id}/")]
        if paths:
            self._blob.delete_blob(paths)

    def read_upload(self, pathname: str) -> bytes:
        try:
            return self._blob.get_blob(pathname, access=self.access).body
        except self._blob.BlobNotFoundError as exc:
            raise FileNotFoundError(pathname) from exc

    def delete_upload(self, pathname: str) -> None:
        try:
            self._blob.delete_blob(pathname)
        except Exception:  # cleanup is best effort; never mask the ingest result
            log.warning("Could not delete temporary upload %s", pathname)
