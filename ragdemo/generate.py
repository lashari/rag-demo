"""Answer generation grounded in retrieved context, with inline citations.

- ClaudeGenerator:     fluent answers from Claude grounded in context
                       (needs ANTHROPIC_API_KEY and the `anthropic` package).
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


def _user_prompt(question: str, results) -> str:
    return f"Context:\n{_format_context(results)}\n\nQuestion: {question}"


class ExtractiveGenerator:
    name = "extractive"

    def generate(self, question: str, results) -> str:
        if not results:
            return "No relevant context found."
        top = results[:2]
        passages = " ".join(f"{ch.text} [{n}]" for n, (_, _, ch) in enumerate(top, 1))
        return (
            "(extractive answer — set ANTHROPIC_API_KEY or OPENAI_API_KEY for a generated one)\n\n" + passages
        )


class OpenAIGenerator:
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI  # lazy import
        self.client = OpenAI()
        self.model = model or config.CHAT_MODEL

    def generate(self, question: str, results) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": _user_prompt(question, results)},
            ],
        )
        return resp.choices[0].message.content.strip()


class ClaudeGenerator:
    """Grounded answers from Claude via the official Anthropic SDK."""

    name = "claude"
    MAX_TOKENS = 16000
    # Models that accept server-side refusal fallbacks ("default" routing).
    FALLBACK_BETA = "server-side-fallback-2026-07-01"
    FALLBACK_MODELS = {
        "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5",
        "claude-fable-5-1", "claude-fable-5-2",
    }
    REFUSAL_MESSAGE = "Claude declined to answer this question."

    def __init__(self, model: str | None = None, effort: str | None = None,
                 client=None) -> None:
        if client is None:
            import anthropic  # lazy import: optional dependency
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.client = client
        self.model = model or config.CLAUDE_MODEL
        self.effort = effort or config.CLAUDE_EFFORT

    def generate(self, question: str, results) -> str:
        kwargs = {}
        if self.model in self.FALLBACK_MODELS:
            # If Claude declines, the API retries on a fallback model in the same call.
            kwargs = {"betas": [self.FALLBACK_BETA], "fallbacks": "default"}
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.MAX_TOKENS,
            system=SYSTEM,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": _user_prompt(question, results)}],
            **kwargs,
        )
        if resp.stop_reason == "refusal":
            return self.REFUSAL_MESSAGE
        return "".join(b.text for b in resp.content if b.type == "text").strip()


def get_generator():
    """Claude if ANTHROPIC_API_KEY is set, else OpenAI if OPENAI_API_KEY is set, else offline."""
    if config.has_anthropic():
        return ClaudeGenerator()
    if config.has_openai():
        return OpenAIGenerator()
    return ExtractiveGenerator()
