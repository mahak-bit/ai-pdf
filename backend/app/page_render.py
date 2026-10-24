"""Render a PDF page to PNG, optionally highlighting a cited passage on it."""

from pathlib import Path

import pymupdf

from .errors import NotFound

# Search the page for short word windows: PyMuPDF's search is line-sensitive, so matching a
# whole multi-line passage at once fails whenever the text wraps differently.
_WINDOW_WORDS = 6
_HIGHLIGHT_RGB = (0.69, 0.78, 1.0)


def _locate(page: pymupdf.Page, passage: str) -> list[pymupdf.Quad]:
    words = passage.split()
    quads: list[pymupdf.Quad] = []
    for start in range(0, len(words), _WINDOW_WORDS):
        fragment = " ".join(words[start : start + _WINDOW_WORDS])
        if len(fragment) >= 8:
            quads.extend(page.search_for(fragment, quads=True))
    return quads


def render_page_png(pdf_path: Path, page_number: int, highlight: str | None = None, zoom: float = 1.6) -> bytes:
    with pymupdf.open(pdf_path) as doc:
        if not 1 <= page_number <= doc.page_count:
            raise NotFound(f"Page {page_number} does not exist in this document.")
        page = doc[page_number - 1]
        if highlight:
            # Annotations live only on this in-memory copy; the stored PDF is never modified.
            for quad in _locate(page, highlight):
                annot = page.add_highlight_annot(quad)
                annot.set_colors(stroke=_HIGHLIGHT_RGB)
                annot.update()
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), annots=True)
        return pixmap.tobytes("png")
