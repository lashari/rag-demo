"""Document loading + sentence-aware chunking with overlap."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class Chunk:
    id: str
    text: str
    source: str


def split_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split text into overlapping chunks, preferring to break on sentence boundaries."""
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:  # try not to cut mid-sentence
            dot = text.rfind(". ", start + int(size * 0.6), end)
            if dot != -1:
                end = dot + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def load_documents(path: str) -> list[Chunk]:
    """Load .md/.txt files from a file or directory into overlapping chunks."""
    p = Path(path)
    files = [p] if p.is_file() else sorted(
        f for f in p.rglob("*") if f.suffix.lower() in {".md", ".txt"}
    )
    chunks: list[Chunk] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        for i, c in enumerate(split_text(text)):
            chunks.append(Chunk(id=f"{f.name}#{i}", text=c, source=f.name))
    return chunks
