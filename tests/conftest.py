"""Shared test setup: every test runs against the offline backends."""
import pytest

from ragdemo import config


@pytest.fixture(autouse=True)
def no_api_keys(monkeypatch):
    # Ignore any keys from a local .env so tests never call a real API.
    # Tests that need a backend selected set a fake key themselves.
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
