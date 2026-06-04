"""Unit tests for the RAG layer.

No real AWS calls happen: the bedrock client factories are monkeypatched
with MagicMocks. These tests pin the behaviour that matters for a RAG
system — grounded answers cite their sources, and weak retrieval is
refused before any generation tokens are spent.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app import rag


def _retrieve_response(score: float):
    return {
        "retrievalResults": [
            {
                "content": {"text": "Pillars: operational excellence and security."},
                "location": {
                    "type": "S3",
                    "s3Location": {"uri": "s3://corpus/well-architected.pdf"},
                },
                "score": score,
            }
        ]
    }


def _converse_response(text: str):
    return {"output": {"message": {"content": [{"text": text}]}}}


def test_grounded_answer_returns_citations(monkeypatch):
    agent = MagicMock()
    agent.retrieve.return_value = _retrieve_response(0.82)
    runtime = MagicMock()
    runtime.converse.return_value = _converse_response(
        "The pillars are ... [source: well-architected.pdf]"
    )
    monkeypatch.setattr(rag, "_agent_client", lambda: agent)
    monkeypatch.setattr(rag, "_runtime_client", lambda: runtime)

    result = rag.answer_question("What are the Well-Architected pillars?")

    assert result.grounded is True
    assert result.citations == ["well-architected.pdf"]
    runtime.converse.assert_called_once()


def test_low_score_question_is_refused(monkeypatch):
    """Below-threshold retrieval -> refuse, and never call generation."""
    agent = MagicMock()
    agent.retrieve.return_value = _retrieve_response(0.10)  # < MIN_RELEVANCE_SCORE
    runtime = MagicMock()
    monkeypatch.setattr(rag, "_agent_client", lambda: agent)
    monkeypatch.setattr(rag, "_runtime_client", lambda: runtime)

    result = rag.answer_question("What is the boiling point of helium?")

    assert result.grounded is False
    assert result.text == rag.REFUSAL_MESSAGE
    assert result.citations == []
    runtime.converse.assert_not_called()


def test_no_results_is_refused(monkeypatch):
    agent = MagicMock()
    agent.retrieve.return_value = {"retrievalResults": []}
    runtime = MagicMock()
    monkeypatch.setattr(rag, "_agent_client", lambda: agent)
    monkeypatch.setattr(rag, "_runtime_client", lambda: runtime)

    result = rag.answer_question("anything at all")

    assert result.grounded is False
    runtime.converse.assert_not_called()


def test_source_parsed_from_s3_uri():
    location = {"s3Location": {"uri": "s3://bucket/folder/guide.pdf"}}
    assert rag._source_from_location(location) == "guide.pdf"
    assert rag._source_from_location({}) == "unknown"
