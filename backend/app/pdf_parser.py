"""PDF validation and page-preserving text extraction (PyMuPDF)."""

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

from .errors import InvalidUpload, UnprocessableDocument

# Below this many extractable characters a document is treated as having no text layer.
MIN_DOCUMENT_CHARS = 40
# A page with fewer characters than this but with images is likely a scanned page.
MIN_PAGE_CHARS = 10

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_LINE_BREAKS = re.compile(r"\s*\n\s*")
_DIGITS = re.compile(r"\d+")
# Running headers/footers: short blocks repeated on at least half the pages (min 3).
_BOILERPLATE_MAX_CHARS = 120


@dataclass
class PageText:
    number: int  # 1-based, as printed by PDF viewers
    text: str


@dataclass
class ParsedDocument:
    pages: list[PageText]
    page_count: int
    title: str | None = None
    image_only_pages: list[int] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)


def validate_upload(filename: str | None, data: bytes, max_bytes: int) -> None:
    """Cheap checks that run before the PDF is opened."""
    if not filename or not filename.lower().endswith(".pdf"):
        raise InvalidUpload("Only PDF files are supported. Choose a file ending in .pdf.")
    if not data:
        raise InvalidUpload("The uploaded file is empty.")
    if len(data) > max_bytes:
        limit_mb = max_bytes // (1024 * 1024)
        raise InvalidUpload(
            f"The file is larger than the {limit_mb} MB limit.",
            code="file_too_large",
            status_code=413,
        )
    # The PDF header may be preceded by junk bytes; the spec allows it within the first 1 KB.
    if b"%PDF-" not in data[:1024]:
        raise InvalidUpload("This file is not a valid PDF (missing PDF header).")


def _clean_block(text: str) -> str:
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _LINE_BREAKS.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def _boilerplate_key(paragraph: str) -> str:
    # Page numbers differ between pages, so compare with digits masked.
    return _DIGITS.sub("#", paragraph.lower())


def _find_boilerplate(pages: list[list[str]]) -> set[str]:
    """Keys of short paragraphs that repeat across many pages (running headers/footers)."""
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for paragraphs in pages:
        counts.update({_boilerplate_key(p) for p in paragraphs if len(p) <= _BOILERPLATE_MAX_CHARS})
    threshold = max(3, math.ceil(len(pages) / 2))
    return {key for key, n in counts.items() if n >= threshold}


def extract_pages(data: bytes, max_pages: int) -> ParsedDocument:
    """Open a PDF from bytes and return its text page by page.

    Raises UnprocessableDocument for corrupted, encrypted, empty or image-only PDFs.
    """
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several exception types for broken files
        raise UnprocessableDocument(
            "The PDF could not be opened. It may be corrupted.", code="corrupted_pdf"
        ) from exc

    with doc:
        if doc.needs_pass:
            raise UnprocessableDocument(
                "This PDF is password-protected. Remove the password and upload it again.",
                code="encrypted_pdf",
            )
        if doc.page_count == 0:
            raise UnprocessableDocument("The PDF has no pages.", code="empty_pdf")
        if doc.page_count > max_pages:
            raise UnprocessableDocument(
                f"The PDF has {doc.page_count} pages; the limit is {max_pages}.",
                code="too_many_pages",
            )

        page_paragraphs: list[list[str]] = []
        has_images: list[bool] = []
        for page in doc:
            blocks = page.get_text("blocks", sort=True)
            page_paragraphs.append([p for p in (_clean_block(b[4]) for b in blocks if b[6] == 0) if p])
            has_images.append(bool(page.get_images(full=False)))

        title = (doc.metadata or {}).get("title") or None

    boilerplate = _find_boilerplate(page_paragraphs)
    pages: list[PageText] = []
    image_only: list[int] = []
    for index, paragraphs in enumerate(page_paragraphs):
        text = "\n\n".join(p for p in paragraphs if _boilerplate_key(p) not in boilerplate)
        if len(text) < MIN_PAGE_CHARS and has_images[index]:
            image_only.append(index + 1)
        pages.append(PageText(number=index + 1, text=text))

    parsed = ParsedDocument(
        pages=pages, page_count=len(pages), title=title, image_only_pages=image_only
    )
    if parsed.char_count < MIN_DOCUMENT_CHARS:
        if image_only:
            raise UnprocessableDocument(
                "This PDF looks scanned: its pages are images with no text layer. "
                "Run it through OCR (for example, ocrmypdf) and upload the result.",
                code="scanned_pdf",
            )
        raise UnprocessableDocument(
            "No readable text was found in this PDF.", code="no_text"
        )
    return parsed
