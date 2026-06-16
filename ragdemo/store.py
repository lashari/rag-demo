"""Minimal in-memory cosine vector store.

Production note: I run this on Qdrant (HNSW index + int8 quantization) at ~12M
vectors; an in-memory NumPy store keeps this demo zero-setup while showing the
same search semantics (L2-normalize, then cosine = dot product).
"""
from __future__ import annotations

import numpy as np


class VectorStore:
    def __init__(self) -> None:
        self._vecs: np.ndarray | None = None
        self._payloads: list = []

    def add(self, vectors: np.ndarray, payloads: list) -> None:
        v = vectors.astype(np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        v = v / norms
        self._vecs = v if self._vecs is None else np.vstack([self._vecs, v])
        self._payloads.extend(payloads)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[int, float, object]]:
        if self._vecs is None:
            return []
        q = query_vec.astype(np.float32).ravel()
        q = q / (np.linalg.norm(q) or 1.0)
        sims = self._vecs @ q  # cosine (stored vectors are normalized)
        order = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i]), self._payloads[i]) for i in order]

    def __len__(self) -> int:
        return 0 if self._vecs is None else len(self._payloads)
