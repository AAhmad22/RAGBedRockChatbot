"""Unit tests for the evaluation harness.

The live pipeline and the judge model are mocked, so scoring logic is tested
without any AWS calls.
"""

from __future__ import annotations

import eval.run_eval as ev
from app.rag import Answer


def test_evaluate_all_correct(monkeypatch):
    rows = [
        {"question": "in scope", "ground_truth": "some fact"},
        {"question": "off topic", "ground_truth": "OUT_OF_SCOPE"},
    ]
    answers = {
        "in scope": Answer(text="grounded reply", citations=["doc.pdf"], grounded=True),
        "off topic": Answer(text=ev.OUT_OF_SCOPE, citations=[], grounded=False),
    }
    monkeypatch.setattr(ev, "answer_question", lambda q: answers[q])
    monkeypatch.setattr(ev, "judge_relevance", lambda q, a: 5.0)

    m = ev.evaluate(rows)

    assert m.total == 2
    assert m.guardrail_accuracy == 1.0
    assert m.citation_rate == 1.0
    assert m.mean_relevance == 5.0


def test_guardrail_penalises_hallucination(monkeypatch):
    # Out-of-scope question that the model wrongly answers -> guardrail miss.
    rows = [{"question": "off topic", "ground_truth": "OUT_OF_SCOPE"}]
    monkeypatch.setattr(
        ev,
        "answer_question",
        lambda q: Answer(text="made up", citations=[], grounded=True),
    )
    monkeypatch.setattr(ev, "judge_relevance", lambda q, a: 1.0)

    m = ev.evaluate(rows)

    assert m.guardrail_accuracy == 0.0


def test_citation_rate_counts_only_grounded(monkeypatch):
    rows = [
        {"question": "q1", "ground_truth": "x"},
        {"question": "q2", "ground_truth": "y"},
    ]
    answers = {
        "q1": Answer(text="a", citations=["s.pdf"], grounded=True),
        "q2": Answer(text="b", citations=[], grounded=True),
    }
    monkeypatch.setattr(ev, "answer_question", lambda q: answers[q])
    monkeypatch.setattr(ev, "judge_relevance", lambda q, a: 4.0)

    m = ev.evaluate(rows)

    assert m.citation_rate == 0.5  # 1 of 2 grounded answers had a citation
