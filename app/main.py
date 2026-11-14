"""FastAPI service exposing the agent over HTTP.

Demonstrates API development practices: typed request/response models,
a health check endpoint for ops/monitoring, and structured responses that
surface routing/cost/latency metadata (useful for dashboards and debugging).
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent import Agent, AgentResult
from app.lcel_chain import (
    build_assistant_chain,
    build_assistant_chain_streaming,
    build_tool_calling_chain,
    build_tool_calling_chain_streaming,
    build_tool_calling_chain_streaming_with_memory,
    build_tool_calling_chain_with_memory,
)
from app.react_agent import ReActResult
from app.react_langgraph_agent import ReActLangGraphAgent
from app.reasoning import generate_with_reasoning, stream_with_reasoning
from app.reasoning_langgraph_agent import ReasoningLangGraphAgent
from app.reasoning_react_agent import ReasoningReActAgent, ReasoningReActResult


def _warm_up_vector_indexes() -> None:
    """Pre-build the codebase/README FAISS indexes in a background thread as
    soon as the server starts, instead of leaving them fully lazy.

    Both ``app.codebase_rag`` (whole-project source code) and
    ``app.readme_rag`` (README.md) build their FAISS index by making ONE
    Ollama embedding call per chunk, sequentially, the first time their
    'search' tool is actually used. For the codebase index specifically
    (~100 chunks across the whole project), that means the very first
    real user question that needs ``codebase_search`` pays the full
    embedding-build cost synchronously, inside that request -- which is
    exactly the kind of "it just hung for ages" experience we want to avoid.

    Kicking the build off here, in a background thread, means it's usually
    already finished (or well underway) by the time a real user asks a
    code question, without blocking server startup or any individual
    request on it. If Ollama isn't reachable yet (or at all), the build
    simply fails silently here -- the SAME lazy build-on-first-use fallback
    in ``app.codebase_rag``/``app.readme_rag`` still applies as before, so
    nothing is broken, this is a pure latency optimization.

    Skipped entirely under pytest (``PYTEST_CURRENT_TEST`` is set by pytest
    itself) so the test suite never makes real network calls to Ollama just
    from importing/constructing the FastAPI app.
    """
    if os.environ.get("PYTEST_CURRENT_TEST") is not None:
        return

    def _build() -> None:
        try:
            from app.codebase_rag import _default_codebase_index

            _default_codebase_index.search("warm up the index", k=1)
        except Exception:
            pass

        try:
            from app.readme_rag import _default_readme_index

            _default_readme_index.search("warm up the index", k=1)
        except Exception:
            pass

    threading.Thread(target=_build, daemon=True, name="vector-index-warmup").start()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _warm_up_vector_indexes()
    yield


app = FastAPI(
    title="GenAI Agent Demo",
    description=(
        "Demo service showing LLM agent orchestration, model routing by "
        "cost/latency, and failure-mode mitigation."
    ),
    version="1.0.0",
    lifespan=_lifespan,
)


_agent = Agent()
_react_langgraph_agent = ReActLangGraphAgent()
_reasoning_react_agent = ReasoningReActAgent()
_reasoning_langgraph_agent = ReasoningLangGraphAgent()
# Built once at import time -- building it just composes Runnables (cheap,
# no network call), so there's no need to rebuild it per-request.
_lcel_chain = build_assistant_chain()
# NOT built eagerly -- constructing it is cheap, but it requires Ollama
# running with a tool-calling-capable model to actually be usable, so it's
# built lazily on first request to /chat/lcel/tools instead.
_lcel_tool_chain = None
_lcel_tool_chain_with_memory = None
_lcel_tool_chain_streaming = None
_lcel_tool_chain_streaming_with_memory = None


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


def _get_lcel_tool_chain_streaming():
    global _lcel_tool_chain_streaming
    if _lcel_tool_chain_streaming is None:
        _lcel_tool_chain_streaming = build_tool_calling_chain_streaming()
    return _lcel_tool_chain_streaming


def _get_lcel_tool_chain_streaming_with_memory():
    global _lcel_tool_chain_streaming_with_memory
    if _lcel_tool_chain_streaming_with_memory is None:
        _lcel_tool_chain_streaming_with_memory = build_tool_calling_chain_streaming_with_memory()
    return _lcel_tool_chain_streaming_with_memory


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
    thread_id: str | None = Field(
        default=None,
        description=(
            "Used by /agent/react and /agent/react/reasoning/langgraph. An "
            "arbitrary id identifying this conversation -- the server "
            "remembers every turn under the same thread_id (via LangGraph's "
            "MemorySaver checkpointer) and automatically includes it on the "
            "next call with that same id, so (like 'session_id' above, but "
            "for the LangGraph agents) the caller does NOT need to resend "
            "prior turns via 'history' themselves. Use a different thread_id "
            "per independent conversation/user. Omit it to fall back to "
            "stateless, caller-managed 'history' behavior."
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


class ReasoningChatResponse(BaseModel):
    reasoning: str = Field(
        ..., description="The model's chain-of-thought, shown separately from its final answer."
    )
    answer: str
    model: str
    latency_ms: int


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
    awaiting_user_input: bool = Field(
        default=False,
        description=(
            "True if the agent called 'ask_user' and stopped -- 'answer' is "
            "then a QUESTION for the real user, not a final answer. Show it "
            "to them, collect their reply, and call this endpoint again "
            "with that reply as 'prompt' (same history/thread_id as before) "
            "to resume where the agent left off."
        ),
    )

    @classmethod
    def from_react_result(cls, result: ReActResult) -> "ReActResponse":
        return cls(
            answer=result.answer,
            steps=[ReActStepResponse(**s.__dict__) for s in result.steps],
            iterations=result.iterations,
            model=result.model,
            hit_iteration_limit=result.hit_iteration_limit,
            awaiting_user_input=result.awaiting_user_input,
        )


class ReasoningReActStepResponse(BaseModel):
    tool_name: str
    tool_args: dict
    observation: str
    reasoning: str


class ReasoningReActResponse(BaseModel):
    answer: str
    reasoning: str = Field(
        ..., description="The chain-of-thought behind the FINAL answer/turn."
    )
    steps: list[ReasoningReActStepResponse]
    iterations: int
    model: str
    hit_iteration_limit: bool
    awaiting_user_input: bool = Field(
        default=False,
        description=(
            "Same meaning as ReActResponse.awaiting_user_input -- True if "
            "'ask_user' stopped the loop, in which case 'answer' is a "
            "question for the real user, not a final answer."
        ),
    )

    @classmethod
    def from_result(cls, result: ReasoningReActResult) -> "ReasoningReActResponse":
        return cls(
            answer=result.answer,
            reasoning=result.reasoning,
            steps=[ReasoningReActStepResponse(**s.__dict__) for s in result.steps],
            iterations=result.iterations,
            model=result.model,
            hit_iteration_limit=result.hit_iteration_limit,
            awaiting_user_input=result.awaiting_user_input,
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


@app.post("/chat/lcel/stream")
def chat_lcel_stream(request: ChatRequest) -> StreamingResponse:
    """Streaming counterpart to /chat/lcel: the response body arrives as
    plain-text chunks as they're generated by the model (via
    ``app.lcel_chain.build_assistant_chain_streaming``'s ``.stream()``)
    instead of waiting for the whole answer before responding -- this is
    what actually fixes the "I have to wait a lot" latency-to-first-byte
    problem, since the client sees output start arriving almost immediately."""
    chain = build_assistant_chain_streaming()

    def _generate():
        yield from chain.stream({"user_request": request.prompt})

    return StreamingResponse(_generate(), media_type="text/plain")


@app.post("/chat/lcel/tools/stream")
def chat_lcel_tools_stream(request: ChatRequest) -> StreamingResponse:
    """Streaming counterpart to /chat/lcel/tools, via
    ``app.lcel_chain.build_tool_calling_chain_streaming``. Requires Ollama
    running locally with a tool-calling-capable model."""
    chain = _get_lcel_tool_chain_streaming()

    def _generate():
        yield from chain.stream({"user_request": request.prompt})

    return StreamingResponse(_generate(), media_type="text/plain")


@app.post("/chat/lcel/tools/memory/stream")
def chat_lcel_tools_memory_stream(request: ChatRequest) -> StreamingResponse:
    """Streaming + server-side memory, combining /chat/lcel/tools/stream with
    /chat/lcel/tools/memory, via
    ``app.lcel_chain.build_tool_calling_chain_streaming_with_memory``."""
    chain = _get_lcel_tool_chain_streaming_with_memory()

    def _generate():
        yield from chain.stream(
            {"user_request": request.prompt},
            config={"configurable": {"session_id": request.session_id}},
        )

    return StreamingResponse(_generate(), media_type="text/plain")


@app.post("/agent/react", response_model=ReActResponse)
def agent_react(request: ChatRequest) -> ReActResponse:
    """Same task as /chat, but the LLM itself decides which tool(s) to call
    and when to stop, instead of our own routing rules. Requires Ollama
    running locally with a tool-calling-capable model (e.g. llama3.1).

    Orchestrated as a LangGraph ``StateGraph`` via
    ``app.react_langgraph_agent.ReActLangGraphAgent``, compiled with
    LangGraph's ``MemorySaver`` checkpointer. Pass ``thread_id`` to get
    server-managed multi-turn memory: the server snapshots the full
    conversation state (including real past tool calls/results, not just
    prior answer text) after every turn and retrieves it automatically on
    the next call with the SAME thread_id -- you don't need to resend
    ``history`` yourself, and a file path (or any other value learned via a
    tool call) earlier in the conversation is never silently lost even if a
    past answer was vague. Omit ``thread_id`` and pass prior turns via
    ``history`` instead for the original, caller-managed behavior."""
    history = [{"role": m.role, "content": m.content} for m in request.history]
    result = _react_langgraph_agent.handle(
        request.prompt, history=history, thread_id=request.thread_id
    )
    return ReActResponse.from_react_result(result)


@app.post("/agent/react/langgraph", response_model=ReActResponse)
def agent_react_langgraph(request: ChatRequest) -> ReActResponse:
    """Deprecated alias for /agent/react -- both now serve the same
    ``ReActLangGraphAgent``, kept around only for backwards compatibility
    with existing callers of this specific path. Prefer /agent/react."""
    return agent_react(request)


@app.post("/agent/react/reasoning", response_model=ReasoningReActResponse)
def agent_react_reasoning(request: ChatRequest) -> ReasoningReActResponse:
    """Like /agent/react, but for a REASONING model (DeepSeek-R1 by default
    -- see ``app.config.OLLAMA_REASONING_MODEL``) via
    ``app.reasoning_react_agent.ReasoningReActAgent``. Ollama's native
    tool-calling JSON mechanism (used by /agent/react) is NOT reliable for
    this model, so this endpoint uses a different, TEXT-based ReAct protocol
    instead (the model emits plain ``Action:``/``Action Input:`` lines,
    parsed and executed by us) -- verified empirically to work reliably.
    See the module docstring in ``app.reasoning_react_agent`` for the exact
    tests. Each step's chain-of-thought is included in the response so you
    can see WHY the model decided to call (or not call) a given tool."""
    history = [{"role": m.role, "content": m.content} for m in request.history]
    result = _reasoning_react_agent.handle(request.prompt, history=history)
    return ReasoningReActResponse.from_result(result)


@app.post("/agent/react/reasoning/langgraph", response_model=ReasoningReActResponse)
def agent_react_reasoning_langgraph(request: ChatRequest) -> ReasoningReActResponse:
    """Same task and response shape as /agent/react/reasoning (same system
    prompt, tools, and text-based Action/Action Input protocol), but
    orchestrated as an explicit LangGraph ``StateGraph`` via
    ``app.reasoning_langgraph_agent.ReasoningLangGraphAgent`` instead of a
    plain Python for-loop. The key difference: a genuinely malformed tool
    call (invalid JSON in Action Input) is caught by a conditional graph
    edge that routes back to the model with the REAL parse error as
    feedback -- an explicit self-correction cycle, instead of folding retry
    logic into one long method. See that module's docstring for the graph
    shape and rationale.

    Pass ``thread_id`` to get server-managed multi-turn memory via
    LangGraph's ``MemorySaver`` checkpointer -- the server remembers every
    turn sent under that id automatically, so you don't need to resend
    ``history`` yourself (same idea as ``session_id`` for
    /chat/lcel/tools/memory). Omit ``thread_id`` to fall back to the
    original, caller-managed ``history`` behavior."""
    history = [{"role": m.role, "content": m.content} for m in request.history]
    result = _reasoning_langgraph_agent.handle(
        request.prompt, history=history, thread_id=request.thread_id
    )
    return ReasoningReActResponse.from_result(result)


@app.post("/chat/reasoning", response_model=ReasoningChatResponse)
def chat_reasoning(request: ChatRequest) -> ReasoningChatResponse:
    """A dedicated REASONING-model path (DeepSeek-R1 by default -- see
    ``app.config.OLLAMA_REASONING_MODEL``), separate from the fast/balanced/
    powerful cost-tier routing used by /chat. Returns the model's visible
    chain-of-thought (``reasoning``) SEPARATELY from its final answer, via
    ``app.reasoning.generate_with_reasoning``. Requires Ollama running
    locally with a reasoning-capable model pulled (e.g.
    ``ollama pull deepseek-r1:14b``). Reasoning models can legitimately take
    much longer than a normal completion -- that's expected, not a hang."""
    result = generate_with_reasoning(request.prompt)
    return ReasoningChatResponse(
        reasoning=result.reasoning,
        answer=result.answer,
        model=result.model,
        latency_ms=result.latency_ms,
    )


@app.post("/chat/reasoning/stream")
def chat_reasoning_stream(request: ChatRequest) -> StreamingResponse:
    """Streaming counterpart to /chat/reasoning. Each line of the response
    body is a small JSON object ``{"kind": "reasoning"|"answer", "text": ...}``
    (JSON Lines / NDJSON) so the client can render the chain-of-thought and
    the final answer in separate UI areas AS THEY STREAM IN, instead of only
    being able to tell them apart after the whole response is done."""

    def _generate():
        for kind, text in stream_with_reasoning(request.prompt):
            yield json.dumps({"kind": kind, "text": text}) + "\n"

    return StreamingResponse(_generate(), media_type="application/x-ndjson")


