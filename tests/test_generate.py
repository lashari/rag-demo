"""Offline tests for answer generation (ragdemo/generate.py).

The OpenAI path is exercised with a fake `openai` module, so no API key,
network access or real client is needed.
"""
import sys
import types

import pytest

from ragdemo import config, generate
from ragdemo.chunk import Chunk


def results(*texts):
    """Build retriever-style results: (id, score, Chunk)."""
    return [
        (f"doc{n}.md#0", 1.0 / n, Chunk(id=f"doc{n}.md#0", text=t, source=f"doc{n}.md"))
        for n, t in enumerate(texts, 1)
    ]


class FakeCompletions:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = types.SimpleNamespace(content=self.reply)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


@pytest.fixture
def fake_openai(monkeypatch):
    """Install a fake `openai` module; returns the completions recorder."""
    completions = FakeCompletions(reply="  RRF merges ranked lists [1].  \n")

    class FakeOpenAI:
        instances = 0

        def __init__(self):
            FakeOpenAI.instances += 1
            self.chat = types.SimpleNamespace(completions=completions)

    module = types.ModuleType("openai")
    module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    completions.client_class = FakeOpenAI
    return completions


# --- context formatting -----------------------------------------------------

def test_format_context_numbers_passages_with_sources():
    ctx = generate._format_context(results("Alpha text.", "Beta text."))
    assert ctx == (
        "[1] (source: doc1.md) Alpha text.\n\n"
        "[2] (source: doc2.md) Beta text."
    )


def test_format_context_empty():
    assert generate._format_context([]) == ""


# --- extractive (offline) generator ----------------------------------------

def test_extractive_no_results():
    assert generate.ExtractiveGenerator().generate("q", []) == "No relevant context found."


def test_extractive_cites_top_two_passages_only():
    out = generate.ExtractiveGenerator().generate("q", results("One.", "Two.", "Three."))
    assert out.startswith("(extractive answer")
    assert "One. [1] Two. [2]" in out
    assert "Three." not in out


# --- OpenAI generator with a fake client ------------------------------------

def test_openai_generator_sends_grounded_prompt(fake_openai, monkeypatch):
    monkeypatch.setattr(config, "CHAT_MODEL", "test-model")
    gen = generate.OpenAIGenerator()
    answer = gen.generate("What is RRF?", results("RRF fuses rankings.", "BM25 is lexical."))

    assert answer == "RRF merges ranked lists [1]."  # reply is stripped
    assert gen.name == "openai"
    assert len(fake_openai.calls) == 1

    call = fake_openai.calls[0]
    assert call["model"] == "test-model"
    assert call["temperature"] == 0.2
    system, user = call["messages"]
    assert system == {"role": "system", "content": generate.SYSTEM}
    assert user["role"] == "user"
    assert user["content"] == (
        "Context:\n"
        "[1] (source: doc1.md) RRF fuses rankings.\n\n"
        "[2] (source: doc2.md) BM25 is lexical.\n\n"
        "Question: What is RRF?"
    )


def test_openai_generator_model_argument_overrides_config(fake_openai, monkeypatch):
    monkeypatch.setattr(config, "CHAT_MODEL", "from-config")
    generate.OpenAIGenerator(model="explicit-model").generate("q", results("x"))
    assert fake_openai.calls[0]["model"] == "explicit-model"


def test_openai_generator_with_no_results_still_asks(fake_openai):
    generate.OpenAIGenerator().generate("q", [])
    assert fake_openai.calls[0]["messages"][1]["content"] == "Context:\n\n\nQuestion: q"


# --- backend selection ------------------------------------------------------

def test_get_generator_offline_without_keys():
    assert isinstance(generate.get_generator(), generate.ExtractiveGenerator)


def test_get_generator_uses_openai_when_key_set(fake_openai, monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    gen = generate.get_generator()
    assert isinstance(gen, generate.OpenAIGenerator)
    assert fake_openai.client_class.instances == 1
