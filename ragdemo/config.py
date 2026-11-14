"""Configuration via environment variables (loads a local .env if present)."""
from __future__ import annotations
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # python-dotenv is optional
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "medium")  # low | medium | high | xhigh | max

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))
CANDIDATES = int(os.getenv("CANDIDATES", "10"))
HASH_DIM = int(os.getenv("HASH_DIM", "512"))


def has_openai() -> bool:
    """True when an OpenAI key is configured (enables real embeddings + generation)."""
    return bool(OPENAI_API_KEY)


def has_anthropic() -> bool:
    """True when an Anthropic key is configured (enables Claude-generated answers)."""
    return bool(ANTHROPIC_API_KEY)
