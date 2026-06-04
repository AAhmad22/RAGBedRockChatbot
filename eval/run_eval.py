"""Evaluation harness for the RAG chatbot.

Loads eval/dataset.jsonl, runs each question through the pipeline, and scores
faithfulness, answer relevance, and context precision. The OUT_OF_SCOPE rows
verify the "I don't know" guardrail fires correctly.

TODO: wire up RAGAS (https://docs.ragas.io) or implement lightweight
LLM-as-judge scoring with Bedrock. Print a results table and write it to
eval/results.md so it can be committed and shown in the README.
"""

from __future__ import annotations

import json
from pathlib import Path

DATASET = Path(__file__).parent / "dataset.jsonl"


def load_dataset() -> list[dict]:
    return [json.loads(line) for line in DATASET.read_text().splitlines() if line]


def main() -> None:
    rows = load_dataset()
    print(f"Loaded {len(rows)} eval examples.")
    raise NotImplementedError("Implement scoring (RAGAS or LLM-as-judge).")


if __name__ == "__main__":
    main()
