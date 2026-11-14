"""Offline tests for the Claude generator (ragdemo.generate.ClaudeGenerator).

Most tests inject a fake client. One test drives the real `anthropic` SDK
against a mock HTTP transport, so the request shape is checked against the
actual library without any network access.
"""
import json
import sys
import types

import pytest

from ragdemo import config, generate
from ragdemo.chunk import Chunk


def results(*texts):
    return [
        (f"doc{n}.md#0", 1.0 / n, Chunk(id=f"doc{n}.md#0", text=t, source=f"doc{n}.md"))
        for n, t in enumerate(texts, 1)
    ]


def text_block(text):
    return types.SimpleNamespace(type="text", text=text)


class FakeClaudeClient:
    """Records beta.messages.create calls and returns a canned message."""

    def __init__(self, content=None, stop_reason="end_turn"):
        self.calls = []
        self.content = content if content is not None else [text_block("RRF fuses rankings [1].")]
        self.stop_reason = stop_reason
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(content=self.content, stop_reason=self.stop_reason)


# --- request shape ----------------------------------------------------------

def test_sends_grounded_prompt_with_defaults(monkeypatch):
    monkeypatch.setattr(config, "CLAUDE_MODEL", "claude-opus-5-5")
    monkeypatch.setattr(config, "CLAUDE_EFFORT", "medium")
    client = FakeClaudeClient()
    gen = generate.ClaudeGenerator(client=client)

    answer = gen.generate("What is RRF?", results("RRF fuses rankings.", "BM25 is lexical."))

    assert answer == "RRF fuses rankings [1]."
    assert gen.name == "claude"
    (call,) = client.calls
    assert call["model"] == "claude-opus-5-5"
    assert call["max_tokens"] == 16000
    assert call["system"] == generate.SYSTEM
    assert call["output_config"] == {"effort": "medium"}
    assert call["messages"] == [{
        "role": "user",
        "content": (
            "Context:\n"
            "[1] (source: doc1.md) RRF fuses rankings.\n\n"
            "[2] (source: doc2.md) BM25 is lexical.\n\n"
            "Question: What is RRF?"
        ),
    }]
    # No sampling params: current Claude models reject temperature/top_p/top_k.
    assert not {"temperature", "top_p", "top_k"} & call.keys()


def test_model_and_effort_arguments_override_config():
    client = FakeClaudeClient()
    generate.ClaudeGenerator(model="claude-sonnet-5-5", effort="low", client=client).generate(
        "q", results("x"))
    assert client.calls[0]["model"] == "claude-sonnet-5-5"
    assert client.calls[0]["output_config"] == {"effort": "low"}


@pytest.mark.parametrize("model", sorted(generate.ClaudeGenerator.FALLBACK_MODELS))
def test_refusal_fallbacks_enabled_for_supported_models(model):
    client = FakeClaudeClient()
    generate.ClaudeGenerator(model=model, client=client).generate("q", results("x"))
    assert client.calls[0]["betas"] == ["server-side-fallback-2026-07-01"]
    assert client.calls[0]["fallbacks"] == "default"


def test_no_fallbacks_for_other_models():
    client = FakeClaudeClient()
    generate.ClaudeGenerator(model="claude-haiku-5-5", client=client).generate("q", results("x"))
    assert "fallbacks" not in client.calls[0]
    assert "betas" not in client.calls[0]


# --- response handling ------------------------------------------------------

def test_joins_text_blocks_and_skips_other_block_types():
    content = [
        types.SimpleNamespace(type="thinking", thinking=""),
        text_block("Part one [1]. "),
        types.SimpleNamespace(type="fallback"),
        text_block("Part two [2].  "),
    ]
    gen = generate.ClaudeGenerator(client=FakeClaudeClient(content=content))
    assert gen.generate("q", results("a", "b")) == "Part one [1]. Part two [2]."


def test_refusal_returns_clear_message():
    client = FakeClaudeClient(content=[], stop_reason="refusal")
    gen = generate.ClaudeGenerator(client=client)
    assert gen.generate("q", results("x")) == generate.ClaudeGenerator.REFUSAL_MESSAGE


def test_creates_sdk_client_lazily(monkeypatch):
    created = []

    class FakeAnthropic:
        def __init__(self):
            created.append(self)

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=FakeAnthropic))
    gen = generate.ClaudeGenerator()
    assert len(created) == 1 and gen.client is created[0]


# --- backend selection ------------------------------------------------------

def test_get_generator_prefers_claude_when_both_keys_set(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=object))
    assert isinstance(generate.get_generator(), generate.ClaudeGenerator)


def test_has_anthropic_follows_key(monkeypatch):
    assert not config.has_anthropic()
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-ant-test")
    assert config.has_anthropic()


# --- real SDK, mock transport -----------------------------------------------

def test_real_sdk_request_and_response():
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx2.Response(200, json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": "Fused answer [1]."}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })

    client = anthropic.Anthropic(
        api_key="sk-ant-test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    gen = generate.ClaudeGenerator(model="claude-opus-5-5", effort="medium", client=client)

    assert gen.generate("What is RRF?", results("RRF fuses rankings.")) == "Fused answer [1]."

    (request,) = seen
    assert request.url.path == "/v1/messages"
    assert "server-side-fallback-2026-07-01" in request.headers["anthropic-beta"]
    body = json.loads(request.content)
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 16000
    assert body["system"] == generate.SYSTEM
    assert body["output_config"] == {"effort": "medium"}
    assert body["fallbacks"] == "default"
    assert body["messages"][0]["content"].endswith("Question: What is RRF?")
