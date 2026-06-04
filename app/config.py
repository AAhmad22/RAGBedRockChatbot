"""Runtime configuration, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    knowledge_base_id: str = os.getenv("KNOWLEDGE_BASE_ID", "")
    generation_model_id: str = os.getenv(
        "GENERATION_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"
    )
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    # Minimum retrieval score below which we refuse to answer (guardrail).
    min_relevance_score: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.4"))


settings = Settings()
