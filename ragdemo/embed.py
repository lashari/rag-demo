"""Pluggable embedders.

- OpenAIEmbedder: real semantic embeddings (needs OPENAI_API_KEY).
- HashEmbedder:   deterministic, dependency-free bag-of-words hashing so the
                  demo runs fully offline. Not semantic — good enough to show
                  the pipeline working end-to-end without any API key.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from . import config

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


class HashEmbedder:
    name = "hash"

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or config.HASH_DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in _tokenize(text):
                idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
                out[row, idx] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI  # lazy import — only needed in live mode
        self.client = OpenAI()
        self.model = model or config.EMBED_MODEL

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=list(texts))
        return np.array([d.embedding for d in resp.data], dtype=np.float32)


def get_embedder():
    """Return OpenAI embedder when a key is set, otherwise the offline hash embedder."""
    return OpenAIEmbedder() if config.has_openai() else HashEmbedder()
