"""FastAPI service exposing the agent over HTTP.

Demonstrates API development practices: typed request/response models,
a health check endpoint for ops/monitoring, and structured responses that
surface routing/cost/latency metadata (useful for dashboards and debugging).
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.agent import Agent, AgentResult

app = FastAPI(
    title="GenAI Agent Demo",
    description=(
        "Demo service showing LLM agent orchestration, model routing by "
        "cost/latency, and failure-mode mitigation."
    ),
    version="1.0.0",
)

_agent = Agent()


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User's natural language request.")


class ChatResponse(BaseModel):
    answer: str
    tool_used: str | None
    model: str | None
    tier: str | None
    latency_ms: int
    estimated_cost_usd: float
    retries: int
    degraded: bool
    routing_reason: str

    @classmethod
    def from_agent_result(cls, result: AgentResult) -> "ChatResponse":
        return cls(**result.__dict__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = _agent.handle(request.prompt)
    return ChatResponse.from_agent_result(result)
