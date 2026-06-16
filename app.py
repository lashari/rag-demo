"""FastAPI service for the hybrid RAG demo.

Run:  uvicorn app:app --reload
Then: POST /query {"question": "..."}  ·  GET /health  ·  docs at /docs
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from ragdemo.pipeline import RagPipeline

app = FastAPI(title="Hybrid RAG Demo", version="1.0.0")
rag = RagPipeline()


@app.on_event("startup")
def _startup() -> None:
    rag.ingest_path(str(Path(__file__).parent / "data"))


class Query(BaseModel):
    question: str
    top_k: int | None = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "ready": rag.ready,
        "embedder": rag.embedder.name,
        "generator": rag.generator.name,
    }


@app.post("/query")
def query(q: Query) -> dict:
    return rag.answer(q.question, top_k=q.top_k)
