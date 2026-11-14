"""A LangGraph reimplementation of ``app.react_agent``'s native Ollama
``tool_calls`` ReAct loop, adding REAL multi-turn chat history via a
LangGraph ``Checkpointer`` instead of requiring the caller to resend
``history`` (or this app's ``known_files`` workaround) on every request.

Why a Checkpointer, and why this fixes the "forgot the file path" bug
-----------------------------------------------------------------------
``app.react_agent.ReActAgent.handle()`` is caller-managed: on every call the
caller must pass the FULL prior conversation back in via ``history``, and
even then only each past turn's plain ANSWER TEXT is kept -- the raw
``tool_calls``/arguments behind a turn are deliberately dropped (see that
module's ``_extract_touched_files`` docstring for the bug this caused: a
file path was unrecoverable on a later turn if the model's own answer was
vague, e.g. "I've saved it for you").

A LangGraph ``Checkpointer`` (``MemorySaver`` here; swap in
``SqliteSaver``/``PostgresSaver`` for a store that survives a restart) takes
a snapshot of the graph's ENTIRE state -- including the full ``messages``
list, with real ``tool_calls`` and ``tool`` observation messages intact --
after every node execution, keyed by a ``thread_id``. Calling
``handle(prompt, thread_id="some-id")`` again:
    1. Looks up ``thread_id``'s last saved state (``self._graph.get_state``).
    2. Appends the new user message to that saved ``messages`` list.
    3. Invokes the graph, which persists the UPDATED conversation back under
       the same ``thread_id`` automatically.
A different (or omitted) ``thread_id`` starts from a blank slate. Because
the raw tool-call messages are part of what's checkpointed, a file path
(or any other value learned via a tool call) is never silently lost the way
it was in the plain-Python-loop version -- no extra ``known_files``
bookkeeping is needed for this to work.

Graph shape::

    START -> agent --[no tool_calls => final answer]--> END
               |
               `--[tool_calls present]--> execute_tools --> agent (loop)

Same system prompt, tool schemas, and native ``tool_calls`` protocol as
``app.react_agent.ReActAgent`` (reuses its ``_build_tool_schemas``,
``execute_tool_call``, and ``_extract_touched_files`` directly) -- this
module only changes HOW multi-turn memory and the loop are orchestrated.
"""
from __future__ import annotations

import json
import uuid
from typing import TypedDict

import httpx
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.config import MODEL_CATALOG, OLLAMA_BASE_URL
from app.prompts import DEFAULT_COMPANY_NAME, render_react_system_prompt
from app.react_agent import (
    ASK_USER_TOOL_NAME,
    MAX_ITERATIONS,
    OLLAMA_NUM_CTX,
    REACT_TIMEOUT_SECONDS,
    SEARCH_TOOL_NAMES,
    ReActResult,
    ReActStep,
    _build_tool_schemas,
    _extract_touched_files,
    _known_files_message,
    execute_tool_call,
    is_thin_search_result,
    search_escalation_hint,
    search_relevance_hint,
)
from app.tools import AVAILABLE_TOOLS


class _GraphState(TypedDict, total=False):
    messages: list[dict]
    steps: list[ReActStep]
    iterations: int
    pending_tool_calls: list[dict] | None
    answer: str | None
    hit_iteration_limit: bool
    # Deterministic search-escalation tracking (see app.react_agent's
    # search_escalation_hint docstring for the full rationale) -- a plain
    # list (not a set) since graph state must stay JSON/pickle-friendly for
    # the MemorySaver checkpointer.
    tried_search_tools: list[str]
    # True once an 'ask_user' tool call has short-circuited the loop (see
    # app.tools.ask_user's docstring and ASK_USER_TOOL_NAME's comment in
    # app.react_agent) -- in that case ``answer`` holds the QUESTION to show
    # the real user, not a genuine final answer.
    awaiting_user_input: bool


class ReActLangGraphAgent:
    """Same native tool-calling capability as ``ReActAgent``, orchestrated
    as a LangGraph ``StateGraph`` compiled with a ``MemorySaver``
    checkpointer for server-managed, ``thread_id``-keyed multi-turn memory
    (see module docstring)."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str = OLLAMA_BASE_URL,
        company_name: str = DEFAULT_COMPANY_NAME,
    ) -> None:
        self._model = model or MODEL_CATALOG["balanced"].ollama_model
        self._base_url = base_url.rstrip("/")
        self._tool_schemas = _build_tool_schemas()
        self._system_prompt = render_react_system_prompt(
            tool_names=[t.name for t in AVAILABLE_TOOLS],
            company_name=company_name,
        )
        self._memory = MemorySaver()
        self._graph = self._build_graph()

    def _call_model(self, messages: list[dict]) -> dict:
        response = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "tools": self._tool_schemas,
                "stream": False,
                "options": {"num_ctx": OLLAMA_NUM_CTX},
            },
            timeout=REACT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    def _call_model_stream(self, messages: list[dict]):
        """Streaming counterpart to ``_call_model`` (same request shape as
        ``app.react_agent.ReActAgent._call_model_stream``) -- yields each raw
        NDJSON chunk Ollama sends as it's generated, instead of waiting for
        the whole response. Used by ``handle_stream`` below, which bypasses
        the compiled graph's own node-by-node execution (that only runs
        non-streaming ``_call_model``) so a UI can show live thinking/content
        deltas while STILL reading/writing through this agent's checkpointer
        for ``thread_id`` continuity."""
        with httpx.stream(
            "POST",
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "tools": self._tool_schemas,
                "stream": True,
                "options": {"num_ctx": OLLAMA_NUM_CTX},
            },
            timeout=REACT_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                yield json.loads(line)

    def _resolve_initial_messages(
        self,
        prompt: str,
        config: dict,
        history: list[dict] | None,
        known_files: list[str] | None,
        thread_id: str | None,
    ) -> list[dict]:
        """Shared by ``handle`` and ``handle_stream``: load the base
        conversation either from the checkpointer (``thread_id`` given) or
        from caller-managed ``history``/``known_files`` (unchanged,
        ``ReActAgent``-compatible fallback when ``thread_id`` is omitted)."""
        if thread_id is not None:
            snapshot = self._graph.get_state(config)
            prior_messages = (snapshot.values or {}).get("messages") if snapshot else None
            if prior_messages:
                return [*prior_messages, {"role": "user", "content": prompt}]

        messages = [{"role": "system", "content": self._system_prompt}]
        if history:
            messages.extend(history)
        if known_files:
            messages.append(_known_files_message(known_files))
        messages.append({"role": "user", "content": prompt})
        return messages

    def _agent_node(self, state: _GraphState) -> _GraphState:
        iterations = state.get("iterations", 0) + 1
        if iterations > MAX_ITERATIONS:
            return {
                **state,
                "iterations": iterations,
                "hit_iteration_limit": True,
                "answer": (
                    "I wasn't able to settle on a final answer within the "
                    f"allowed number of steps ({MAX_ITERATIONS})."
                ),
            }

        data = self._call_model(state["messages"])
        message = data.get("message", {})
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            # Final answer -- append the FULL assistant turn to messages so
            # the checkpoint captures it (otherwise a later call on the same
            # thread_id would see the user's prompt but not this reply).
            return {
                **state,
                "iterations": iterations,
                "answer": message.get("content", ""),
                "messages": state["messages"]
                + [{"role": "assistant", "content": message.get("content", "")}],
            }

        # BUG FIX carried over from app.react_agent.ReActAgent.handle(): drop
        # the "thinking" field before persisting this assistant turn -- qwen3
        # emits chain-of-thought even without --think, and replaying it back
        # verbatim measurably confused the model on the VERY NEXT turn (see
        # that module's long comment for the live-tested details).
        assistant_msg = {
            "role": "assistant",
            "content": message.get("content", ""),
            "tool_calls": tool_calls,
        }
        return {
            **state,
            "iterations": iterations,
            "messages": state["messages"] + [assistant_msg],
            "pending_tool_calls": tool_calls,
        }

    def _execute_tools_node(self, state: _GraphState) -> _GraphState:
        messages = list(state["messages"])
        steps = list(state.get("steps", []))
        tried_search_tools = set(state.get("tried_search_tools") or [])
        for call in state.get("pending_tool_calls") or []:
            step = execute_tool_call(call)
            steps.append(step)
            tool_message = {"role": "tool", "content": step.observation}
            if call.get("id"):
                tool_message["tool_call_id"] = call["id"]
            messages.append(tool_message)

            # Human-in-the-loop short-circuit -- see ASK_USER_TOOL_NAME's
            # comment in app.react_agent / app.tools.ask_user's docstring.
            # Stop processing any further pending calls in this same batch
            # and surface the question as this turn's "answer" immediately,
            # routed straight to END by _route_after_tools below instead of
            # looping back to the model for a round it has no real reply to.
            if step.tool_name == ASK_USER_TOOL_NAME:
                return {
                    **state,
                    "messages": messages,
                    "steps": steps,
                    "pending_tool_calls": None,
                    "tried_search_tools": sorted(tried_search_tools),
                    "answer": step.tool_args.get("question", step.observation),
                    "awaiting_user_input": True,
                }

            # Deterministic search-escalation nudge -- see
            # app.react_agent.search_escalation_hint's/search_relevance_hint's
            # docstrings.
            if step.tool_name in SEARCH_TOOL_NAMES:
                tried_search_tools.add(step.tool_name)
                if is_thin_search_result(step.observation):
                    hint = search_escalation_hint(step.tool_name, tried_search_tools)
                else:
                    hint = search_relevance_hint(
                        step.tool_name, step.tool_args.get("query", ""), tried_search_tools
                    )
                if hint:
                    messages.append(hint)
        return {
            **state,
            "messages": messages,
            "steps": steps,
            "pending_tool_calls": None,
            "tried_search_tools": sorted(tried_search_tools),
        }

    @staticmethod
    def _route_after_agent(state: _GraphState) -> str:
        if state.get("answer") is not None:
            return "end"
        return "execute_tools"

    @staticmethod
    def _route_after_tools(state: _GraphState) -> str:
        """Normally loops back to 'agent' for another round -- EXCEPT when
        'ask_user' just short-circuited this node (see _execute_tools_node),
        in which case 'answer' is already set and the graph must stop here,
        the same way _route_after_agent stops for a genuine final answer."""
        if state.get("answer") is not None:
            return "end"
        return "agent"

    def _build_graph(self):
        graph = StateGraph(_GraphState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("execute_tools", self._execute_tools_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"end": END, "execute_tools": "execute_tools"},
        )
        graph.add_conditional_edges(
            "execute_tools",
            self._route_after_tools,
            {"end": END, "agent": "agent"},
        )
        return graph.compile(checkpointer=self._memory)

    def handle(
        self,
        prompt: str,
        history: list[dict] | None = None,
        known_files: list[str] | None = None,
        thread_id: str | None = None,
    ) -> ReActResult:
        """Same contract as ``ReActAgent.handle``, plus an optional
        ``thread_id`` for server-managed multi-turn memory via the
        ``MemorySaver`` checkpointer (see module docstring).

        - ``thread_id`` given: the checkpointer's saved conversation for
          that id (if any) -- including real past ``tool_calls``/``tool``
          messages, not just plain answer text -- is used as the base
          history, and ``prompt`` is appended to it. ``history`` and
          ``known_files`` are ignored in this case, since the checkpoint
          already carries everything they'd otherwise be compensating for.
          The updated conversation is persisted back automatically.
        - ``thread_id`` omitted (default): unchanged, caller-managed
          ``history``/``known_files`` behavior, identical to
          ``ReActAgent.handle``.
        """
        config = {
            "recursion_limit": MAX_ITERATIONS * 4 + 10,
            "configurable": {"thread_id": thread_id or uuid.uuid4().hex},
        }
        messages = self._resolve_initial_messages(prompt, config, history, known_files, thread_id)

        final_state = self._graph.invoke(
            {
                "messages": messages,
                "steps": [],
                "iterations": 0,
                "tried_search_tools": [],
                "awaiting_user_input": False,
            },
            config=config,
        )

        steps = final_state.get("steps", [])
        return ReActResult(
            answer=final_state.get("answer") or "",
            steps=steps,
            iterations=final_state.get("iterations", 0),
            model=self._model,
            hit_iteration_limit=final_state.get("hit_iteration_limit", False),
            touched_files=_extract_touched_files(steps),
            awaiting_user_input=final_state.get("awaiting_user_input", False),
        )

    def handle_stream(
        self,
        prompt: str,
        history: list[dict] | None = None,
        known_files: list[str] | None = None,
        thread_id: str | None = None,
    ):
        """Streaming counterpart to ``handle()``, for UIs that want to show
        the model's output as it's generated (e.g. ``app.gradio_app``) while
        STILL getting checkpointer-backed ``thread_id`` memory.

        Drives the same loop as ``app.react_agent.ReActAgent.handle_stream``
        (identical yielded event shapes: ``thinking_delta``, ``content_delta``,
        ``tool_call``, ``observation``, ``final``) directly in Python instead
        of through the compiled graph's node execution -- LangGraph's own
        node-by-node runner only supports non-streaming calls to the model,
        so live deltas require bypassing it here. Continuity is still fully
        checkpointer-backed: the base conversation is loaded via
        ``self._graph.get_state`` (same as ``handle()``), and the FINAL
        conversation (with real ``tool_calls``/``tool`` messages intact) is
        written back via ``self._graph.update_state`` once the turn
        completes, so the NEXT call with the same ``thread_id`` -- whether
        via ``handle()`` or ``handle_stream()`` -- sees it."""
        config = {
            "recursion_limit": MAX_ITERATIONS * 4 + 10,
            "configurable": {"thread_id": thread_id or uuid.uuid4().hex},
        }
        messages = self._resolve_initial_messages(prompt, config, history, known_files, thread_id)
        steps: list[ReActStep] = []
        tried_search_tools: set[str] = set()

        for iteration in range(1, MAX_ITERATIONS + 1):
            content_acc = ""
            tool_calls: list[dict] = []

            for chunk in self._call_model_stream(messages):
                chunk_message = chunk.get("message", {})
                thinking_piece = chunk_message.get("thinking") or ""
                content_piece = chunk_message.get("content") or ""
                if thinking_piece:
                    yield {"type": "thinking_delta", "text": thinking_piece}
                if content_piece:
                    content_acc += content_piece
                    yield {"type": "content_delta", "text": content_piece}
                if chunk_message.get("tool_calls"):
                    tool_calls = chunk_message["tool_calls"]

            if not tool_calls:
                messages = messages + [{"role": "assistant", "content": content_acc}]
                self._graph.update_state(
                    config, {"messages": messages, "steps": steps, "iterations": iteration}
                )
                result = ReActResult(
                    answer=content_acc,
                    steps=steps,
                    iterations=iteration,
                    model=self._model,
                    touched_files=_extract_touched_files(steps),
                )
                yield {"type": "final", "result": result}
                return

            # Same "don't replay thinking" fix as app.react_agent -- see that
            # module's long comment for the live-tested rationale.
            messages = messages + [
                {"role": "assistant", "content": content_acc, "tool_calls": tool_calls}
            ]
            for call in tool_calls:
                fn = call.get("function", {})
                yield {
                    "type": "tool_call",
                    "tool_name": fn.get("name", ""),
                    "tool_args": fn.get("arguments", {}),
                }
                step = execute_tool_call(call)
                steps.append(step)
                yield {
                    "type": "observation",
                    "tool_name": step.tool_name,
                    "observation": step.observation,
                }
                tool_message = {"role": "tool", "content": step.observation}
                if call.get("id"):
                    tool_message["tool_call_id"] = call["id"]
                messages = messages + [tool_message]

                # Human-in-the-loop short-circuit -- same rationale as
                # ``_execute_tools_node``/app.react_agent's handle(). Persist
                # the conversation so far (including this question) back to
                # the checkpointer, then stop -- the next call with the SAME
                # thread_id, once the caller supplies the user's reply as the
                # new prompt, resumes exactly where this left off.
                if step.tool_name == ASK_USER_TOOL_NAME:
                    self._graph.update_state(
                        config,
                        {
                            "messages": messages,
                            "steps": steps,
                            "iterations": iteration,
                            "awaiting_user_input": True,
                        },
                    )
                    result = ReActResult(
                        answer=step.tool_args.get("question", step.observation),
                        steps=steps,
                        iterations=iteration,
                        model=self._model,
                        touched_files=_extract_touched_files(steps),
                        awaiting_user_input=True,
                    )
                    yield {"type": "final", "result": result}
                    return

                if step.tool_name in SEARCH_TOOL_NAMES:
                    tried_search_tools.add(step.tool_name)
                    if is_thin_search_result(step.observation):
                        hint = search_escalation_hint(step.tool_name, tried_search_tools)
                    else:
                        hint = search_relevance_hint(
                            step.tool_name, step.tool_args.get("query", ""), tried_search_tools
                        )
                    if hint:
                        messages = messages + [hint]

        self._graph.update_state(
            config,
            {
                "messages": messages,
                "steps": steps,
                "iterations": MAX_ITERATIONS,
                "hit_iteration_limit": True,
            },
        )
        result = ReActResult(
            answer=(
                "I wasn't able to settle on a final answer within the allowed "
                f"number of steps ({MAX_ITERATIONS})."
            ),
            steps=steps,
            iterations=MAX_ITERATIONS,
            model=self._model,
            hit_iteration_limit=True,
            touched_files=_extract_touched_files(steps),
        )
        yield {"type": "final", "result": result}

