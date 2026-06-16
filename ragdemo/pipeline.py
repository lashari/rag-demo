"""End-to-end RAG pipeline: ingest -> hybrid retrieve -> grounded answer."""
from __future__ import annotations

from . import config
from .chunk import load_documents
from .embed import get_embedder
from .generate import get_generator
from .retrieve import HybridRetriever


class RagPipeline:
    def __init__(self) -> None:
        self.embedder = get_embedder()
        self.retriever = HybridRetriever(self.embedder)
        self.generator = get_generator()
        self.ready = False

    def ingest_path(self, path: str) -> dict:
        chunks = load_documents(path)
        self.retriever.index(chunks)
        self.ready = True
        return {"chunks_indexed": len(chunks), "embedder": self.embedder.name}

    def answer(self, question: str, top_k: int | None = None) -> dict:
        top_k = top_k or config.TOP_K
        results = self.retriever.search(question, top_k=top_k, candidates=config.CANDIDATES)
        answer = self.generator.generate(question, results)
        sources = [
            {
                "n": n,
                "source": ch.source,
                "score": round(score, 4),
                "preview": ch.text[:160] + ("…" if len(ch.text) > 160 else ""),
            }
            for n, (_, score, ch) in enumerate(results, 1)
        ]
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "generator": self.generator.name,
        }
