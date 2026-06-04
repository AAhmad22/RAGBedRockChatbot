"""FastAPI application exposing the chatbot, also wrapped for AWS Lambda."""

from __future__ import annotations

from fastapi import FastAPI
from mangum import Mangum
from pydantic import BaseModel

from app.rag import answer_question

app = FastAPI(title="RAGBedRockChatbot")


class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, object]:
    result = answer_question(req.question)
    return {
        "answer": result.text,
        "citations": result.citations,
        "grounded": result.grounded,
    }


# AWS Lambda entry point (API Gateway proxy integration).
handler = Mangum(app)
