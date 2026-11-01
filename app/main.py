"""FastAPI service exposing the agent over HTTP.

Demonstrates API development practices: typed request/response models,
a health check endpoint for ops/monitoring, and structured responses that
surface routing/cost/latency metadata (useful for dashboards and debugging).
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.agent import Agent, AgentResult
from app.lcel_chain import (
    build_assistant_chain,
    build_tool_calling_chain,
    build_tool_calling_chain_with_memory,
)
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
# Built once at import time -- building it just composes Runnables (cheap,
# no network call), so there's no need to rebuild it per-request.
_lcel_chain = build_assistant_chain()
# NOT built eagerly -- constructing it is cheap, but it requires Ollama
# running with a tool-calling-capable model to actually be usable, so it's
# built lazily on first request to /chat/lcel/tools instead.
_lcel_tool_chain = None
_lcel_tool_chain_with_memory = None


def _get_lcel_tool_chain():
    global _lcel_tool_chain
    if _lcel_tool_chain is None:
        _lcel_tool_chain = build_tool_calling_chain()
    return _lcel_tool_chain


def _get_lcel_tool_chain_with_memory():
    global _lcel_tool_chain_with_memory
    if _lcel_tool_chain_with_memory is None:
        _lcel_tool_chain_with_memory = build_tool_calling_chain_with_memory()
    return _lcel_tool_chain_with_memory


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'.")
    content: str


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User's natural language request.")
    history: list[ChatMessage] = Field(
        default_factory=list,
        description=(
            "Prior turns of this conversation (role/content pairs, oldest first). "
            "Only used by /agent/react -- without it, each call is stateless and "
            "the model has no memory of anything said earlier."
        ),
    )
    session_id: str = Field(
        default="default",
        description=(
            "Only used by /chat/lcel/tools/memory. An arbitrary id identifying "
            "this conversation -- the server remembers every turn sent under the "
            "same session_id (via LangChain's InMemoryChatMessageHistory) and "
            "automatically includes it on the next call with that same id, so "
            "(unlike 'history' above) the caller does NOT need to resend prior "
            "turns themselves. Use a different session_id per independent "
            "conversation/user."
        ),
    )


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


class LcelChatResponse(BaseModel):
    answer: str


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


@app.post("/chat/lcel", response_model=LcelChatResponse)
def chat_lcel(request: ChatRequest) -> LcelChatResponse:
    """Same plain-LLM task as /chat's non-tool path, but run through the
    declarative LCEL chain from ``app.lcel_chain.build_assistant_chain``
    (``prompt | model | output_parser``) instead of ``Agent.handle()``'s
    imperative routing. No tool-use, routing metadata, or cost/latency
    telemetry -- just the rendered prompt sent straight to the model and
    the parsed text answer, to demonstrate calling an LCEL ``Runnable``
    end-to-end."""
    answer = _lcel_chain.invoke({"user_request": request.prompt})
    return LcelChatResponse(answer=answer)


@app.post("/chat/lcel/tools", response_model=LcelChatResponse)
def chat_lcel_tools(request: ChatRequest) -> LcelChatResponse:
    """Same idea as /agent/react (LLM-driven tool use), but built as a real
    LCEL chain via ``app.lcel_chain.build_tool_calling_chain`` --
    ``some_chat_model.bind_tools(AVAILABLE_TOOLS)`` -- instead of
    ``ReActAgent``'s hand-rolled HTTP loop against Ollama's API. Requires
    Ollama running locally with a tool-calling-capable model."""
    answer = _get_lcel_tool_chain().invoke({"user_request": request.prompt})
    return LcelChatResponse(answer=answer)


@app.post("/chat/lcel/tools/memory", response_model=LcelChatResponse)
def chat_lcel_tools_memory(request: ChatRequest) -> LcelChatResponse:
    """Same as /chat/lcel/tools, but with real conversational memory via
    ``app.lcel_chain.build_tool_calling_chain_with_memory`` -- LangChain's
    PREBUILT ``InMemoryChatMessageHistory`` + ``RunnableWithMessageHistory``.
    Pass the SAME ``session_id`` across requests to continue a conversation;
    unlike /agent/react's ``history`` field, you do NOT need to resend prior
    turns yourself -- the server remembers them for you, keyed by
    ``session_id``. Requires Ollama running locally with a tool-calling-
    capable model."""
    answer = _get_lcel_tool_chain_with_memory().invoke(
        {"user_request": request.prompt},
        config={"configurable": {"session_id": request.session_id}},
    )
    return LcelChatResponse(answer=answer)


@app.post("/agent/react", response_model=ReActResponse)
def agent_react(request: ChatRequest) -> ReActResponse:
    """Same task as /chat, but the LLM itself decides which tool(s) to call
    and when to stop, instead of our own routing rules. Requires Ollama
    running locally with a tool-calling-capable model (e.g. llama3.1).
    Pass prior turns via ``history`` to continue a multi-turn conversation --
    otherwise each call is independent and the model has no memory of
    anything said earlier."""
    history = [{"role": m.role, "content": m.content} for m in request.history]
    result = _react_agent.handle(request.prompt, history=history)
    return ReActResponse.from_react_result(result)
