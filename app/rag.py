"""Core RAG logic.

Two-stage flow:

1. ``retrieve`` against the Bedrock Knowledge Base to get scored chunks.
2. A relevance **guardrail**: if nothing is retrieved, or the best chunk
   scores below ``MIN_RELEVANCE_SCORE``, we refuse *before* calling a
   generation model — so out-of-scope questions never burn generation
   tokens and never get a hallucinated answer.
3. Otherwise, ``converse`` with our own citation-enforcing system prompt,
   passing the retrieved chunks as context.

(An alternative is the single-call ``retrieve_and_generate`` API, which is
simpler but gives less control over the prompt and the score threshold.
The two-stage approach here makes the guardrail explicit and testable.)

The boto3 clients are created via small factory functions so they can be
monkeypatched in unit tests — no real AWS calls happen in CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import boto3

from app.config import settings
from app.prompts import SYSTEM_PROMPT

REFUSAL_MESSAGE = (
    "I don't have enough information in the knowledge base to answer that."
)


@dataclass
class RetrievedChunk:
    """A single chunk returned by the Knowledge Base retriever."""

    text: str
    source: str
    score: float


@dataclass
class Answer:
    """The final answer returned to the caller."""

    text: str
    citations: list[str]
    grounded: bool


def _agent_client() -> Any:
    """Client for Knowledge Base retrieval (bedrock-agent-runtime)."""
    return boto3.client("bedrock-agent-runtime", region_name=settings.aws_region)


def _runtime_client() -> Any:
    """Client for model invocation (bedrock-runtime)."""
    return boto3.client("bedrock-runtime", region_name=settings.aws_region)


def _source_from_location(location: dict[str, Any]) -> str:
    """Turn a retrieval ``location`` into a human-friendly source name.

    Bedrock returns e.g. ``{"type": "S3", "s3Location": {"uri":
    "s3://bucket/path/well-architected.pdf"}}``. We surface just the file
    name for citations.
    """
    uri = location.get("s3Location", {}).get("uri", "")
    return uri.rsplit("/", 1)[-1] if uri else "unknown"


def retrieve(question: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks for ``question`` from the KB."""
    response = _agent_client().retrieve(
        knowledgeBaseId=settings.knowledge_base_id,
        retrievalQuery={"text": question},
        retrievalConfiguration={
            "vectorSearchConfiguration": {"numberOfResults": top_k}
        },
    )
    chunks: list[RetrievedChunk] = []
    for result in response.get("retrievalResults", []):
        chunks.append(
            RetrievedChunk(
                text=result["content"]["text"],
                source=_source_from_location(result.get("location", {})),
                score=float(result.get("score", 0.0)),
            )
        )
    return chunks


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Format chunks into a context block with inline source markers."""
    return "\n\n".join(f"[source: {c.source}]\n{c.text}" for c in chunks)


def generate(question: str, chunks: list[RetrievedChunk]) -> str:
    """Generate a grounded answer from the retrieved context."""
    user_message = f"Context:\n{_build_context(chunks)}\n\nQuestion: {question}"
    response = _runtime_client().converse(
        modelId=settings.generation_model_id,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": user_message}]}],
        inferenceConfig={"temperature": 0.0, "maxTokens": 1024},
    )
    return response["output"]["message"]["content"][0]["text"]


def answer_question(question: str) -> Answer:
    """Full RAG pipeline: retrieve, guardrail, then generate."""
    chunks = retrieve(question)
    top_score = max((c.score for c in chunks), default=0.0)

    # Guardrail: refuse rather than hallucinate when retrieval is weak.
    if not chunks or top_score < settings.min_relevance_score:
        return Answer(text=REFUSAL_MESSAGE, citations=[], grounded=False)

    text = generate(question, chunks)
    citations = sorted({c.source for c in chunks})
    return Answer(text=text, citations=citations, grounded=True)
