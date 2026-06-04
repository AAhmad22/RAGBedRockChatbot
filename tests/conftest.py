"""Shared pytest fixtures. AWS is never called for real in unit tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("KNOWLEDGE_BASE_ID", "TESTKB123")
