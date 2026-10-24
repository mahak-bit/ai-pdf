"""A per-document FAISS index that keeps every vector tied to its source passage.

Vector ids are the chunk's position in ``self.chunks``; ``IndexIDMap2`` stores those ids
explicitly, so a search hit is always resolved through the id, never assumed from order.
"""

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from .chunker import Chunk


@dataclass
class SearchHit:
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]


class DocumentIndex:
    def __init__(self, chunks: list[Chunk], index: faiss.Index):
        if index.ntotal != len(chunks):
            raise ValueError(f"Index holds {index.ntotal} vectors but there are {len(chunks)} chunks")
        self.chunks = chunks
        self.index = index

    @classmethod
    def build(cls, chunks: list[Chunk], vectors: np.ndarray) -> "DocumentIndex":
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk needs exactly one vector")
        if not chunks:
            raise ValueError("Cannot build an index with no chunks")
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
        index.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32), np.arange(len(chunks), dtype=np.int64))
        return cls(chunks, index)

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        k = min(k, len(self.chunks))
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        scores, ids = self.index.search(query, k)
        hits = []
        for score, vector_id in zip(scores[0], ids[0]):
            if vector_id < 0:  # FAISS pads with -1 when fewer results exist
                continue
            hits.append(SearchHit(chunk=self.chunks[int(vector_id)], score=float(score)))
        return hits

    def save(self, path: Path) -> None:
        # serialize_index avoids FAISS's C-level file I/O, which mishandles some Windows paths.
        path.write_bytes(faiss.serialize_index(self.index).tobytes())

    @classmethod
    def load(cls, path: Path, chunks: list[Chunk]) -> "DocumentIndex":
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        index = faiss.deserialize_index(data)
        stored_ids = faiss.vector_to_array(index.id_map) if hasattr(index, "id_map") else None
        if stored_ids is not None and not np.array_equal(np.sort(stored_ids), np.arange(len(chunks))):
            raise ValueError("Stored vector ids do not match the chunk list")
        return cls(chunks, index)
