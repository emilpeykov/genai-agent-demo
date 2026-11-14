"""A LangGraph reimplementation of ``app.reasoning_react_agent``'s text-based
ReAct loop for DeepSeek-R1, with an explicit SELF-CORRECTION graph edge
instead of folding retry-on-bad-output logic into a plain Python loop.

Why rebuild this with LangGraph: DeepSeek-R1, being a reasoning model, often
changes its mind mid-generation or needs a second attempt to produce a clean
tool call. In a straight-line pipeline (like app.lcel_chain's LCEL chains),
a parser failure has nowhere to go but raise or silently degrade -- there's
no cycle. LangGraph's ``StateGraph`` supports CYCLES as first-class edges: a
conditional edge can detect "the last tool call was broken JSON" and route
execution back to the SAME model node with the real parse error appended as
feedback, instead of crashing. ``app.reasoning_react_agent`` already does
something similar ad-hoc inline (its ``skeleton`` retry message for missing
arguments), but folds it into one long method; this module expresses the
same idea as an explicit graph structure, AND additionally surfaces the real
``json.JSONDecodeError`` message back to the model when the JSON itself is
malformed (not just "you forgot the arguments"), so the model's own
"thinking" can diagnose the actual syntax mistake instead of guessing.

Graph shape::

    START -> agent --[final answer / iteration limit]--> END
               |
               |--[tool call, valid args]--> execute_tool --> agent (loop)
               |
               `--[tool call, broken/missing args]--> self_correct -> agent (loop)

Same system prompt, tool list, and text-based ``Action:``/``Action Input:``
protocol as ``app.reasoning_react_agent`` (see that module's docstring for
the empirical tests proving this protocol works for DeepSeek-R1 where
Ollama's native ``tool_calls`` mechanism does not) -- this module only
changes HOW the retry loop is orchestrated, not the protocol itself.

Multi-turn memory via a Checkpointer
-------------------------------------
The graph is compiled with LangGraph's ``MemorySaver`` checkpointer, which
snapshots the full graph state (including ``messages``) after every node
execution and keys those snapshots by a ``thread_id``. Call ``handle(...,
thread_id="some-id")`` and the agent automatically loads that thread's prior
conversation before running, then persists the updated conversation back
under the same id -- no need for the caller to manually collect and resend
``history`` on every call (the way ``/agent/react``'s ``history`` parameter,
still supported here for backwards compatibility, requires). Passing a
DIFFERENT ``thread_id`` (or none at all) starts with a blank history slate.
``MemorySaver`` keeps everything in this process's RAM only -- it does not
survive a restart; swap it for ``SqliteSaver``/``PostgresSaver`` for a
persistent, cross-restart checkpoint store.
"""
from __future__ import annotations

import json
import uuid
from typing import TypedDict

import httpx
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.config import OLLAMA_BASE_URL, OLLAMA_REASONING_MODEL
from app.prompts import DEFAULT_COMPANY_NAME, render_reasoning_react_system_prompt
from app.reasoning_react_agent import (
    ASK_USER_TOOL_NAME,
    MAX_ITERATIONS,
    REASONING_REACT_TIMEOUT_SECONDS,
    ReasoningReActResult,
    ReasoningReActStep,
    _ACTION_RE,
    _FINAL_ANSWER_RE,
    _STOP_SEQUENCES,
    _TOOLS_BY_NAME,
    _build_tool_descriptions,
)


class _GraphState(TypedDict, total=False):
    messages: list[dict]
    steps: list[ReasoningReActStep]
    iterations: int
    pending_name: str | None
    pending_args: dict | None
    pending_error: str | None  # set -> route to self_correct instead of execute_tool
    last_thinking: str
    answer: str | None
    reasoning: str
    hit_iteration_limit: bool
    # See app.react_agent.ReActResult.awaiting_user_input's docstring --
    # same meaning here: set True when 'ask_user' short-circuits the loop,
    # in which case 'answer' holds a QUESTION for the real user.
    awaiting_user_input: bool


class ReasoningLangGraphAgent:
    """Same reasoning + reliable-tool-use capability as
    ``ReasoningReActAgent``, orchestrated as an explicit LangGraph
    self-correcting cycle instead of a plain Python loop (see module
    docstring)."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str = OLLAMA_BASE_URL,
        company_name: str = DEFAULT_COMPANY_NAME,
    ) -> None:
        self._model = model or OLLAMA_REASONING_MODEL
        self._base_url = base_url.rstrip("/")
        self._system_prompt = render_reasoning_react_system_prompt(
            tool_descriptions=_build_tool_descriptions(),
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
                "stream": False,
                "think": True,
                "options": {"stop": _STOP_SEQUENCES},
            },
            timeout=REASONING_REACT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

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
        content = message.get("content", "") or ""
        thinking = message.get("thinking", "") or ""

        action_match = _ACTION_RE.search(content)
        if not action_match:
            final_match = _FINAL_ANSWER_RE.search(content)
            answer = final_match.group("answer").strip() if final_match else content.strip()
            # Append the assistant's final answer to ``messages`` too (not
            # just set "answer") so the MemorySaver checkpoint captures the
            # FULL turn -- otherwise a later call on the same thread_id
            # would see the user's prompt but not the model's reply to it.
            return {
                **state,
                "iterations": iterations,
                "answer": answer,
                "reasoning": thinking,
                "messages": state["messages"] + [{"role": "assistant", "content": answer}],
            }

        name = action_match.group("name")
        raw_args = action_match.group("args")
        truncated_assistant_msg = {"role": "assistant", "content": content[: action_match.end()]}
        base_update = {
            **state,
            "iterations": iterations,
            "messages": state["messages"] + [truncated_assistant_msg],
            "last_thinking": thinking,
        }

        if raw_args is None:
            tool = _TOOLS_BY_NAME.get(name)
            if tool is not None and tool.args:
                skeleton = json.dumps({k: f"<{k}>" for k in tool.args})
                error = (
                    f"no Action Input was given for '{name}', but it requires "
                    "arguments. Re-emit EXACTLY this shape (with real values "
                    f"instead of the <placeholders>):\nAction: {name}\n"
                    f"Action Input: {skeleton}"
                )
                return {**base_update, "pending_name": name, "pending_error": error}
            return {**base_update, "pending_name": name, "pending_args": {}, "pending_error": None}

        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            # THE self-correction edge: a genuinely broken JSON payload is
            # caught here instead of crashing a downstream parser, and the
            # REAL exception message is fed back to the model on the next
            # turn -- so its own "thinking" can diagnose and fix the actual
            # syntax mistake, instead of just being told "try again".
            error = (
                f"the Action Input for '{name}' was not valid JSON ({exc}). "
                f"Re-emit Action: {name} followed by a single valid JSON "
                "object on the Action Input line."
            )
            return {**base_update, "pending_name": name, "pending_error": error}

        return {
            **base_update,
            "pending_name": name,
            "pending_args": args,
            "pending_error": None,
        }

    def _self_correct_node(self, state: _GraphState) -> _GraphState:
        observation_msg = {
            "role": "user",
            "content": f"Observation: Error: {state['pending_error']}",
        }
        return {
            **state,
            "messages": state["messages"] + [observation_msg],
            "pending_name": None,
            "pending_error": None,
        }

    def _execute_tool_node(self, state: _GraphState) -> _GraphState:
        name = state["pending_name"]
        args = state.get("pending_args") or {}
        tool = _TOOLS_BY_NAME.get(name)
        if tool is None:
            observation = f"Error: unknown tool '{name}'"
        else:
            try:
                observation = str(tool.invoke(args))
            except Exception as exc:  # never let a bad tool call crash the graph
                observation = f"Error running tool '{name}': {exc}"

        step = ReasoningReActStep(
            tool_name=name,
            tool_args=args,
            observation=observation,
            reasoning=state.get("last_thinking", ""),
        )
        observation_msg = {"role": "user", "content": f"Observation: {observation}"}

        # Human-in-the-loop short-circuit -- see ASK_USER_TOOL_NAME's comment
        # in app.react_agent / app.tools.ask_user's docstring. Route straight
        # to END (via _route_after_tool below) instead of looping back to
        # the model for a round it has no real reply to yet.
        if name == ASK_USER_TOOL_NAME:
            return {
                **state,
                "steps": [*state.get("steps", []), step],
                "messages": state["messages"] + [observation_msg],
                "pending_name": None,
                "pending_args": None,
                "answer": args.get("question", observation),
                "awaiting_user_input": True,
            }

        return {
            **state,
            "steps": [*state.get("steps", []), step],
            "messages": state["messages"] + [observation_msg],
            "pending_name": None,
            "pending_args": None,
        }

    @staticmethod
    def _route_after_agent(state: _GraphState) -> str:
        if state.get("answer") is not None:
            return "end"
        if state.get("pending_error"):
            return "self_correct"
        return "execute_tool"

    @staticmethod
    def _route_after_tool(state: _GraphState) -> str:
        """Normally loops back to 'agent' -- EXCEPT when 'ask_user' just
        short-circuited _execute_tool_node (``answer`` already set), in
        which case the graph must stop here instead of looping."""
        if state.get("answer") is not None:
            return "end"
        return "agent"

    def _build_graph(self):
        graph = StateGraph(_GraphState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("self_correct", self._self_correct_node)
        graph.add_node("execute_tool", self._execute_tool_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"end": END, "self_correct": "self_correct", "execute_tool": "execute_tool"},
        )
        graph.add_edge("self_correct", "agent")
        graph.add_conditional_edges(
            "execute_tool",
            self._route_after_tool,
            {"end": END, "agent": "agent"},
        )
        return graph.compile(checkpointer=self._memory)

    def handle(
        self,
        prompt: str,
        history: list[dict] | None = None,
        thread_id: str | None = None,
    ) -> ReasoningReActResult:
        """Same contract as ``ReasoningReActAgent.handle``, plus an optional
        ``thread_id`` for server-managed multi-turn memory (see module
        docstring).

        - ``thread_id`` given: the checkpointer's saved conversation for
          that id (if any) is used as the base history, and ``prompt`` is
          appended to it -- ``history`` is ignored in this case, since the
          checkpoint already IS the history. The updated conversation is
          persisted back under the same ``thread_id`` automatically.
        - ``thread_id`` omitted (default): unchanged, caller-managed
          ``history`` behavior -- every call is independent except for
          whatever ``history`` you pass in yourself.
        """
        config = {
            "recursion_limit": MAX_ITERATIONS * 4 + 10,
            "configurable": {"thread_id": thread_id or uuid.uuid4().hex},
        }

        messages: list[dict] | None = None
        if thread_id is not None:
            snapshot = self._graph.get_state(config)
            prior_messages = (snapshot.values or {}).get("messages") if snapshot else None
            if prior_messages:
                messages = [*prior_messages, {"role": "user", "content": prompt}]

        if messages is None:
            messages = [{"role": "system", "content": self._system_prompt}]
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": prompt})

        final_state = self._graph.invoke(
            {"messages": messages, "steps": [], "iterations": 0, "awaiting_user_input": False},
            config=config,
        )

        return ReasoningReActResult(
            answer=final_state.get("answer") or "",
            reasoning=final_state.get("reasoning", ""),
            steps=final_state.get("steps", []),
            iterations=final_state.get("iterations", 0),
            model=self._model,
            hit_iteration_limit=final_state.get("hit_iteration_limit", False),
            awaiting_user_input=final_state.get("awaiting_user_input", False),
        )

