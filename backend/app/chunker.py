"""Page-scoped chunking.

Chunks never cross a page boundary, so every chunk has exactly one page number and a
citation can point at the page the excerpt actually came from. Within a page, sentences
are packed into windows of up to ``max_chars`` with a sentence-aligned overlap.
"""

import re
from dataclasses import asdict, dataclass

from .pdf_parser import PageText

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


@dataclass
class Chunk:
    chunk_id: str  # stable, human-readable: "p{page}-c{n}"
    page: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if paragraph:
            sentences.extend(s.strip() for s in _SENTENCE_END.split(paragraph) if s.strip())
    return sentences


def _hard_wrap(sentence: str, max_chars: int) -> list[str]:
    """Split an over-long sentence (tables, run-on text) on word boundaries."""
    pieces, current = [], ""
    for word in sentence.split():
        if current and len(current) + 1 + len(word) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        pieces.append(current)
    return pieces


def chunk_pages(pages: list[PageText], max_chars: int = 900, overlap_chars: int = 150) -> list[Chunk]:
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    chunks: list[Chunk] = []
    for page in pages:
        units: list[str] = []
        for sentence in split_sentences(page.text):
            units.extend(_hard_wrap(sentence, max_chars) if len(sentence) > max_chars else [sentence])

        window: list[str] = []
        index = 0

        def flush() -> None:
            nonlocal index
            if window:
                chunks.append(Chunk(f"p{page.number}-c{index}", page.number, " ".join(window)))
                index += 1

        for unit in units:
            if window and len(" ".join(window)) + 1 + len(unit) > max_chars:
                flush()
                # Carry trailing sentences forward as overlap, without exceeding the budget.
                carried: list[str] = []
                for prev in reversed(window):
                    if len(" ".join([prev, *carried])) > overlap_chars:
                        break
                    carried.insert(0, prev)
                window = carried if len(" ".join([*carried, unit])) <= max_chars else []
            window.append(unit)
        flush()
    return chunks
