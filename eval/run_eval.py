"""Evaluation harness for the RAG chatbot.

Runs each question in eval/dataset.jsonl through the live pipeline and scores
three things:

  - guardrail accuracy: out-of-scope questions are refused and in-scope ones
    are answered (derived from Answer.grounded; deterministic).
  - citation rate: fraction of grounded answers that include >= 1 source.
  - answer relevance: an LLM-as-judge score (1-5) for in-scope answers.

Results are printed and written to eval/results.md so they can be committed
and shown in the README.

NOTE: unlike the unit tests, this calls real Bedrock (retrieval, generation,
and the judge model), so running it incurs AWS cost. Writing it does not.

Usage:
    python -m eval.run_eval
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import boto3

from app.config import settings
from app.rag import Answer, answer_question

DATASET = Path(__file__).parent / "dataset.jsonl"
RESULTS = Path(__file__).parent / "results.md"
OUT_OF_SCOPE = "OUT_OF_SCOPE"

JUDGE_PROMPT = (
    "You are grading a retrieval-augmented answer. On a scale of 1 to 5, how "
    "well does the ANSWER address the QUESTION? Reply with a single digit "
    "1-5 only.\n\nQUESTION: {question}\nANSWER: {answer}"
)


@dataclass
class Metrics:
    total: int
    guardrail_accuracy: float
    citation_rate: float
    mean_relevance: float


def load_dataset() -> list[dict[str, str]]:
    lines = DATASET.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def judge_relevance(question: str, answer: str) -> float:
    """LLM-as-judge: score how well the answer addresses the question (1-5)."""
    client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
    response = client.converse(
        modelId=settings.generation_model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": JUDGE_PROMPT.format(question=question, answer=answer)}
                ],
            }
        ],
        inferenceConfig={"temperature": 0.0, "maxTokens": 4},
    )
    text = response["output"]["message"]["content"][0]["text"]
    match = re.search(r"[1-5]", text)
    return float(match.group()) if match else 0.0


def evaluate(rows: list[dict[str, str]]) -> Metrics:
    guardrail_correct = 0
    grounded = 0
    cited = 0
    relevance: list[float] = []

    for row in rows:
        question = row["question"]
        expect_refusal = row.get("ground_truth") == OUT_OF_SCOPE
        result: Answer = answer_question(question)

        # Correct when we refuse exactly the out-of-scope questions.
        if result.grounded != expect_refusal:
            guardrail_correct += 1

        if not expect_refusal and result.grounded:
            grounded += 1
            if result.citations:
                cited += 1
            relevance.append(judge_relevance(question, result.text))

    return Metrics(
        total=len(rows),
        guardrail_accuracy=guardrail_correct / len(rows) if rows else 0.0,
        citation_rate=cited / grounded if grounded else 0.0,
        mean_relevance=sum(relevance) / len(relevance) if relevance else 0.0,
    )


def write_results(metrics: Metrics) -> None:
    table = (
        "# Evaluation results\n\n"
        f"Questions evaluated: {metrics.total}\n\n"
        "| Metric | Score |\n"
        "|---|---|\n"
        f"| Guardrail accuracy | {metrics.guardrail_accuracy:.0%} |\n"
        f"| Citation rate | {metrics.citation_rate:.0%} |\n"
        f"| Mean answer relevance (1-5) | {metrics.mean_relevance:.2f} |\n"
    )
    RESULTS.write_text(table)
    print(f"\nWrote results to {RESULTS}")


def main() -> None:
    metrics = evaluate(load_dataset())
    print(f"Guardrail accuracy: {metrics.guardrail_accuracy:.0%}")
    print(f"Citation rate:      {metrics.citation_rate:.0%}")
    print(f"Mean relevance:     {metrics.mean_relevance:.2f}/5")
    write_results(metrics)


if __name__ == "__main__":
    main()
