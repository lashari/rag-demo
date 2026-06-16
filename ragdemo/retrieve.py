"""Hybrid retrieval: dense vector search + BM25 lexical search, merged with
Reciprocal Rank Fusion (RRF).

Why hybrid: dense embeddings give semantic recall ("car" ~ "automobile") while
BM25 gives lexical precision on exact terms, codes, and rare words. RRF fuses
the two rankings without needing to calibrate their (incomparable) score scales.
This mirrors the retrieval stage of the production system I operate.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .store import VectorStore

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60):
    """Fuse several ranked id-lists into one. score(id) = Σ 1/(k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, _id in enumerate(ranking):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores, key=lambda i: -scores[i])
    return ordered, scores


class HybridRetriever:
    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.store = VectorStore()
        self.chunks: list = []
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: list) -> None:
        self.chunks = chunks
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(vectors, list(range(len(chunks))))
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    def search(self, query: str, top_k: int = 4, candidates: int = 10):
        if not self.chunks:
            return []
        # dense ranking (chunk indices)
        qv = self.embedder.embed([query])[0]
        dense = [payload for _, _, payload in self.store.search(qv, candidates)]
        # lexical ranking (chunk indices)
        bm = self._bm25.get_scores(_tokenize(query))
        lexical = sorted(range(len(self.chunks)), key=lambda i: -bm[i])[:candidates]
        # fuse
        fused, scores = reciprocal_rank_fusion([dense, lexical])
        return [(i, scores[i], self.chunks[i]) for i in fused[:top_k]]
