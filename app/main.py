"""FastAPI service exposing the agent over HTTP.

Demonstrates API development practices: typed request/response models,
a health check endpoint for ops/monitoring, and structured responses that
surface routing/cost/latency metadata (useful for dashboards and debugging).
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.agent import Agent, AgentResult
from app.react_agent import ReActAgent, ReActResult

app = FastAPI(
    title="GenAI Agent Demo",
    description=(
        "Demo service showing LLM agent orchestration, model routing by "
        "cost/latency, and failure-mode mitigation."
    ),
    version="1.0.0",
)

_agent = Agent()
_react_agent = ReActAgent()


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


class ReActStepResponse(BaseModel):
    tool_name: str
    tool_args: dict
    observation: str


class ReActResponse(BaseModel):
    answer: str
    steps: list[ReActStepResponse]
    iterations: int
    model: str
    hit_iteration_limit: bool

    @classmethod
    def from_react_result(cls, result: ReActResult) -> "ReActResponse":
        return cls(
            answer=result.answer,
            steps=[ReActStepResponse(**s.__dict__) for s in result.steps],
            iterations=result.iterations,
            model=result.model,
            hit_iteration_limit=result.hit_iteration_limit,
        )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = _agent.handle(request.prompt)
    return ChatResponse.from_agent_result(result)


@app.post("/agent/react", response_model=ReActResponse)
def agent_react(request: ChatRequest) -> ReActResponse:
    """Same task as /chat, but the LLM itself decides which tool(s) to call
    and when to stop, instead of our own routing rules. Requires Ollama
    running locally with a tool-calling-capable model (e.g. llama3.1)."""
    result = _react_agent.handle(request.prompt)
    return ReActResponse.from_react_result(result)
