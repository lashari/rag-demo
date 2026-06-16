"""Answer generation grounded in retrieved context, with inline citations.

- OpenAIGenerator:     fluent answers grounded in context (needs OPENAI_API_KEY).
- ExtractiveGenerator: offline fallback that returns the top passages verbatim
                       with citations, so the demo always produces a cited answer.
"""
from __future__ import annotations

from . import config

SYSTEM = (
    "You are a precise assistant. Answer the question using ONLY the numbered "
    "context below. Cite the sources you use inline like [1], [2]. If the answer "
    "is not in the context, say you don't know — do not invent facts."
)


def _format_context(results) -> str:
    return "\n\n".join(
        f"[{n}] (source: {ch.source}) {ch.text}"
        for n, (_, _, ch) in enumerate(results, 1)
    )


class ExtractiveGenerator:
    name = "extractive"

    def generate(self, question: str, results) -> str:
        if not results:
            return "No relevant context found."
        top = results[:2]
        passages = " ".join(f"{ch.text} [{n}]" for n, (_, _, ch) in enumerate(top, 1))
        return (
            "(extractive answer — set OPENAI_API_KEY for a generated one)\n\n" + passages
        )


class OpenAIGenerator:
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI  # lazy import
        self.client = OpenAI()
        self.model = model or config.CHAT_MODEL

    def generate(self, question: str, results) -> str:
        context = _format_context(results)
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return resp.choices[0].message.content.strip()


def get_generator():
    return OpenAIGenerator() if config.has_openai() else ExtractiveGenerator()
