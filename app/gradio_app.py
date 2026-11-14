"""Gradio chat UI for manually testing the agents without curl.

Run with:
    LLM_BACKEND=ollama .venv/bin/python -m app.gradio_app

Then open the printed local URL (usually http://127.0.0.1:7860) in a browser.
Exposes both agents side-by-side as separate tabs so you can directly compare
the deterministic, rule-based agent against the LLM-driven ReAct agent on the
exact same question.
"""
from __future__ import annotations

import uuid

import gradio as gr

# TEMP: all other agents/chains commented out below -- only the ReAct agent
# (qwen3:14b) tab is active right now. Uncomment the imports/instantiations/
# tabs below to bring the other comparisons back.
# from app.agent import Agent
# from app.lcel_chain import (
#     build_assistant_chain,
#     build_assistant_chain_streaming,
#     build_tool_calling_chain,
#     build_tool_calling_chain_streaming,
#     build_tool_calling_chain_streaming_with_memory,
#     build_tool_calling_chain_with_memory,
# )
from app.react_langgraph_agent import ReActLangGraphAgent
# from app.reasoning import generate_with_reasoning
# from app.reasoning_langgraph_agent import ReasoningLangGraphAgent
# from app.reasoning_react_agent import ReasoningReActAgent

# _rule_based_agent = Agent()
# Hardcoded to qwen3:14b (instead of relying on OLLAMA_BALANCED_MODEL) so this
# tab is self-contained regardless of how the app is launched -- see the two
# live-tested fixes in app.react_agent.ReActAgent.handle() for why qwen3
# specifically needed them (project_dir guidance in the tool schema, and not
# replaying its own "thinking" field back into tool-loop history) -- both
# fixes are shared by ReActLangGraphAgent too, since it reuses the same tool
# schemas/execution helpers from app.react_agent.
#
# Orchestrated as a LangGraph StateGraph with a MemorySaver checkpointer
# (see app.react_langgraph_agent's module docstring) instead of the plain
# Python loop app.react_agent.ReActAgent uses -- multi-turn memory here is
# now handled by a per-browser-tab ``thread_id`` (see ``thread_id_state`` in
# build_app() below) instead of resending ``history`` + the old
# ``known_files`` file-path reminder on every call.
_react_agent = ReActLangGraphAgent(model="qwen3:14b")
# _reasoning_react_agent = ReasoningReActAgent()
# _reasoning_langgraph_agent = ReasoningLangGraphAgent()
# _lcel_chain = build_assistant_chain()
# _lcel_chain_streaming = build_assistant_chain_streaming()
# Not built eagerly -- see app.main for why (requires Ollama to actually use).
# _lcel_tool_chain = None
# _lcel_tool_chain_with_memory = None
# _lcel_tool_chain_streaming = None
# _lcel_tool_chain_streaming_with_memory = None


# def _get_lcel_tool_chain():
#     global _lcel_tool_chain
#     if _lcel_tool_chain is None:
#         _lcel_tool_chain = build_tool_calling_chain()
#     return _lcel_tool_chain
#
#
# def _get_lcel_tool_chain_with_memory():
#     global _lcel_tool_chain_with_memory
#     if _lcel_tool_chain_with_memory is None:
#         _lcel_tool_chain_with_memory = build_tool_calling_chain_with_memory()
#     return _lcel_tool_chain_with_memory
#
#
# def _get_lcel_tool_chain_streaming():
#     global _lcel_tool_chain_streaming
#     if _lcel_tool_chain_streaming is None:
#         _lcel_tool_chain_streaming = build_tool_calling_chain_streaming()
#     return _lcel_tool_chain_streaming
#
#
# def _get_lcel_tool_chain_streaming_with_memory():
#     global _lcel_tool_chain_streaming_with_memory
#     if _lcel_tool_chain_streaming_with_memory is None:
#         _lcel_tool_chain_streaming_with_memory = build_tool_calling_chain_streaming_with_memory()
#     return _lcel_tool_chain_streaming_with_memory

# The separator our own replies use to append debug metadata (tool used,
# cost, etc.) below the actual answer -- see the "meta" strings below. When
# feeding OUR PAST replies back to the model as conversation history, this
# footer must be stripped first, or the model starts seeing (and sometimes
# imitating) its own debug info instead of just the plain answer it gave.
_META_SEPARATOR = "\n\n---\n"

# Invisible (HTML comment, so it doesn't render) marker placed right before
# the REAL final answer in _chat_react's streamed output -- everything
# before it is just the live "thinking"/tool-call/observation transcript,
# which must NOT be replayed back to the model as if it were its own past
# answer (it would see its own <i>thinking</i> HTML and tool-call lines as
# "what it said", and likely imitate or get confused by that, the same
# class of bug documented in app.react_agent around not replaying "thinking"
# within a single request). Only the text AFTER this marker is kept.
_ANSWER_MARKER = "\n<!--answer-->\n"


def _clean_history_for_model(history: list) -> list[dict]:
    """Convert Gradio's chat history into the plain {role, content} messages
    our agents expect, stripping the debug footer we append to replies.

    Without this function, every call to the agent was completely stateless
    -- the model never saw anything said earlier in the conversation, so a
    follow-up like "what about X?" or "ok" made no sense to it.
    """
    cleaned: list[dict] = []
    for turn in history:
        # Gradio 5/6's ChatInterface uses the "messages" format: a flat list
        # of {"role": "user"/"assistant", "content": ...} dicts.
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        if role == "assistant":
            content = content.split(_META_SEPARATOR, 1)[0]
            if _ANSWER_MARKER in content:
                content = content.rsplit(_ANSWER_MARKER, 1)[1]
        cleaned.append({"role": role, "content": content})
    return cleaned


# def _chat_rule_based(message: str, history: list) -> str:
#     result = _rule_based_agent.handle(message)
#     meta = (
#         f"{_META_SEPARATOR}*tool: `{result.tool_used or 'none'}` | "
#         f"model: `{result.model or 'n/a'}` | tier: `{result.tier or 'n/a'}` | "
#         f"cost: ${result.estimated_cost_usd:.5f} | "
#         f"reason: {result.routing_reason}*"
#     )
#     return result.answer + meta


def _react_respond(
    message: str, thinking_history: list, answer_history: list, thread_id: str
):
    """Drives BOTH chatboxes in the custom ReAct tab (see build_app()) from
    a single ``ReActLangGraphAgent.handle_stream`` call, instead of one
    combined growing string like the old ChatInterface-based ``_chat_react``
    did -- ``gr.ChatInterface`` only supports a single output stream, so
    showing "thinking" and the final answer in two SEPARATE chat panels
    requires this hand-rolled gr.Blocks wiring (two gr.Chatbot components,
    both updated together on every yield).

    ``thinking_history``/``answer_history`` are each a list of
    ``{"role", "content"}`` dicts -- gr.Chatbot's "messages" format -- fed
    back in as this function's own prior return value (standard Gradio
    pattern for stateful custom chat UIs). They're PURELY for display here;
    unlike the old ``answer_history``-replay approach, they are NOT resent
    to the model as conversation history on the next call.

    ``thread_id`` is a per-browser-tab ``gr.State`` string (generated once
    when the tab loads -- see build_app() below) identifying this
    conversation to ``ReActLangGraphAgent``'s LangGraph ``MemorySaver``
    checkpointer: the server remembers every past turn (including the REAL
    tool_calls/arguments behind it, e.g. an exact file path) under this id
    and retrieves it automatically on the next call, so unlike the old
    ``known_files``/``answer_history``-replay approach, nothing needs to be
    manually tracked or resent from the UI side at all. See
    ``app.react_langgraph_agent``'s module docstring for why a checkpointer
    fixes the "forgot the file path" bug more completely than that workaround
    did.
    """
    thinking_history = list(thinking_history or [])
    answer_history = list(answer_history or [])

    thinking_history.append({"role": "user", "content": message})
    answer_history.append({"role": "user", "content": message})
    thinking_history.append({"role": "assistant", "content": ""})
    answer_history.append({"role": "assistant", "content": ""})

    transcript = ""  # finalized thinking/tool-call/observation blocks so far
    thinking_acc = ""  # current iteration's live, not-yet-finalized thinking
    content_acc = ""  # current iteration's live content (the final answer, once no more tool calls follow)

    def thinking_view() -> str:
        extra = f"<i>🤔 {thinking_acc}</i>\n\n" if thinking_acc else ""
        return transcript + extra

    for event in _react_agent.handle_stream(message, thread_id=thread_id):
        kind = event["type"]
        if kind == "thinking_delta":
            thinking_acc += event["text"]
            thinking_history[-1]["content"] = thinking_view()
        elif kind == "content_delta":
            content_acc += event["text"]
            answer_history[-1]["content"] = content_acc
        elif kind == "tool_call":
            if thinking_acc:
                transcript += f"<i>🤔 {thinking_acc}</i>\n\n"
                thinking_acc = ""
            content_acc = ""  # discard any stray pre-tool-call content
            transcript += f"🔧 `{event['tool_name']}({event['tool_args']})`\n\n"
            thinking_history[-1]["content"] = thinking_view()
        elif kind == "observation":
            transcript += f"➡️ {event['observation'][:300]}\n\n"
            thinking_history[-1]["content"] = thinking_view()
        elif kind == "final":
            if thinking_acc:
                transcript += f"<i>🤔 {thinking_acc}</i>\n\n"
                thinking_acc = ""
            result = event["result"]
            transcript += (
                f"\n\n---\n*model: `{result.model}` | iterations: {result.iterations} | "
                f"hit iteration limit: {result.hit_iteration_limit}*"
            )
            thinking_history[-1]["content"] = thinking_view()
            if result.awaiting_user_input:
                # 'ask_user' fired: result.answer is a QUESTION, not a final
                # answer, and it never streamed through content_delta (the
                # tool call short-circuits the loop before any more content
                # is generated) -- so content_acc would otherwise be left
                # blank here, making the right-hand panel look stuck/empty
                # even though the agent is correctly waiting for a reply.
                # Show the question clearly and prompt the user to just type
                # their answer as the next message -- thanks to the
                # checkpointer-backed thread_id, that next message naturally
                # resumes this same conversation where it left off.
                answer_history[-1]["content"] = (
                    f"❓ **{result.answer}**\n\n*(Type your reply in the message "
                    "box below to continue.)*"
                )
            else:
                answer_history[-1]["content"] = content_acc
        yield thinking_history, answer_history, "", thread_id


# def _chat_lcel(message: str, history: list) -> str:
#     # Stateless, like /chat/lcel -- the chain has no memory/history input.
#     return _lcel_chain.invoke({"user_request": message})

#
#
# def _chat_lcel_tools(message: str, history: list) -> str:
#     # Stateless, like /chat/lcel/tools. Requires Ollama running locally.
#     return _get_lcel_tool_chain().invoke({"user_request": message})
#
#
# def _chat_lcel_tools_memory(message: str, history: list, session_id: str) -> str:
#     # Like /chat/lcel/tools/memory -- real server-side memory via LangChain's
#     # InMemoryChatMessageHistory, keyed by this browser tab's own session_id
#     # (a gr.State value generated once when the tab first loads, see
#     # build_app() below), so Gradio's own client-side history re-send is NOT
#     # what's providing memory here -- the server remembers on its own.
#     return _get_lcel_tool_chain_with_memory().invoke(
#         {"user_request": message},
#         config={"configurable": {"session_id": session_id}},
#     )
#
#
# def _chat_lcel_stream(message: str, history: list):
#     # Gradio's ChatInterface treats a GENERATOR function specially: each
#     # yielded value replaces the displayed (partial) response so far, giving
#     # a live token-by-token typing effect instead of a single blocking
#     # return -- this is what actually fixes the "wait a long time, then the
#     # whole answer appears at once" latency complaint. We yield the growing
#     # string (not just the new chunk), since that's what Gradio expects.
#     partial = ""
#     for chunk in _lcel_chain_streaming.stream({"user_request": message}):
#         partial += chunk
#         yield partial
#
#
# def _chat_lcel_tools_stream(message: str, history: list):
#     # Streaming version of _chat_lcel_tools. Requires Ollama running locally.
#     partial = ""
#     for chunk in _get_lcel_tool_chain_streaming().stream({"user_request": message}):
#         partial += chunk
#         yield partial
#
#
# def _chat_lcel_tools_memory_stream(message: str, history: list, session_id: str):
#     # Combines _chat_lcel_tools_stream's incremental typing with
#     # _chat_lcel_tools_memory's server-side session memory -- like
#     # /chat/lcel/tools/memory/stream. Same session_id/gr.State pattern as
#     # the memory-only tab (see build_app() below).
#     partial = ""
#     for chunk in _get_lcel_tool_chain_streaming_with_memory().stream(
#         {"user_request": message},
#         config={"configurable": {"session_id": session_id}},
#     ):
#         partial += chunk
#         yield partial
#
#
# def _chat_reasoning(message: str, history: list) -> str:
#     # Stateless, like /chat/reasoning. Requires Ollama running locally with
#     # a reasoning-capable model pulled (default: deepseek-r1:14b -- see
#     # app.config.OLLAMA_REASONING_MODEL). Renders the chain-of-thought in a
#     # collapsed <details> block ABOVE the final answer, so you can inspect
#     # "how" the model got there without it cluttering the main reply.
#     result = generate_with_reasoning(message)
#     reasoning_md = (
#         f"<details><summary>🤔 Show reasoning ({result.latency_ms} ms, "
#         f"model: {result.model})</summary>\n\n{result.reasoning}\n\n</details>\n\n"
#         if result.reasoning
#         else ""
#     )
#     return reasoning_md + result.answer
#
#
# def _render_reasoning_react_result(result) -> str:
#     """Shared rendering for both app.reasoning_react_agent.ReasoningReActAgent
#     and app.reasoning_langgraph_agent.ReasoningLangGraphAgent -- they return
#     the same ReasoningReActResult shape, so the UI presentation is identical."""
#     if result.steps:
#         steps_md = "\n".join(
#             f"{i}. `{s.tool_name}({s.tool_args})` -> {s.observation[:200]}"
#             for i, s in enumerate(result.steps, start=1)
#         )
#     else:
#         steps_md = "(no tools called -- answered directly)"
#     reasoning_md = (
#         f"<details><summary>🤔 Show final reasoning</summary>\n\n{result.reasoning}\n\n</details>\n\n"
#         if result.reasoning
#         else ""
#     )
#     meta = (
#         f"\n\n---\n*model: `{result.model}` | iterations: {result.iterations} | "
#         f"hit iteration limit: {result.hit_iteration_limit}*\n\n**Tool calls:**\n{steps_md}"
#     )
#     return reasoning_md + result.answer + meta
#
#
# def _chat_reasoning_react(message: str, history: list) -> str:
#     # Like /agent/react, but for the reasoning model via a text-based ReAct
#     # protocol instead of Ollama's native (unreliable, for this model)
#     # tool_calls mechanism -- see app.reasoning_react_agent's module
#     # docstring. Stateless w.r.t. conversation history, same as _chat_react.
#     conversation_history = _clean_history_for_model(history)
#     result = _reasoning_react_agent.handle(message, history=conversation_history)
#     return _render_reasoning_react_result(result)
#
#
# def _chat_reasoning_langgraph(message: str, history: list) -> str:
#     # Same task/response shape as _chat_reasoning_react, but orchestrated as
#     # an explicit LangGraph self-correcting cycle instead of a plain Python
#     # loop -- see app.reasoning_langgraph_agent's module docstring.
#     conversation_history = _clean_history_for_model(history)
#     result = _reasoning_langgraph_agent.handle(message, history=conversation_history)

#     return _render_reasoning_react_result(result)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="GenAI Agent Demo") as demo:
        gr.Markdown(
            "# GenAI Agent Demo\n"
            "**All other comparison tabs are temporarily commented out** -- "
            "only the LLM-driven **ReAct agent** (`/agent/react`), hardcoded "
            "to `qwen3:14b`, is active. See `app/gradio_app.py` to restore "
            "the rest (every other tab/handler is still there, just "
            "commented)."
        )
        with gr.Tab("ReAct agent (/agent/react) -- qwen3:14b"):
            gr.Markdown(
                "Two separate panels fed from the SAME live "
                "`ReActLangGraphAgent.handle_stream()` call (see "
                "`_react_respond` in `app/gradio_app.py`) -- `gr.ChatInterface` "
                "only supports one output stream, so showing qwen3's raw "
                "thinking separately from its clean final answer needs this "
                "custom layout.\n\n"
                "- **Left:** live chain-of-thought (*italic*), tool calls, "
                "and real tool observations, as they happen.\n"
                "- **Right:** ONLY the final answer.\n\n"
                "Multi-turn memory is now handled server-side by LangGraph's "
                "`MemorySaver` checkpointer, keyed by a `thread_id` generated "
                "once when this tab loads -- the server remembers every past "
                "turn (including the REAL tool call behind it, e.g. an exact "
                "file path) automatically, so a follow-up like 'read that "
                "file back' works reliably without this UI needing to track "
                "or resend anything itself."
            )
            with gr.Row():
                thinking_box = gr.Chatbot(label="🤔 Thinking & tool calls", height=500)
                answer_box = gr.Chatbot(label="✅ Answer", height=500)
            msg_box = gr.Textbox(
                label="Message",
                placeholder="e.g. how many .py files are in the project dir?",
            )
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")
            gr.Examples(
                examples=[
                    "What is 23 times 17?",
                    "How quickly do you respond to support tickets?",
                    "How much memory is this machine using right now?",
                    "Count the files in the current directory.",
                    "how many .py files are in the project dir?",
                ],
                inputs=msg_box,
            )
            # Hidden per-browser-tab state: a fresh random thread_id, generated
            # once when this tab loads, identifying this conversation to
            # ReActLangGraphAgent's LangGraph MemorySaver checkpointer (see
            # app.react_langgraph_agent's module docstring). Replaces the old
            # known_files/answer_history-replay approach entirely -- the
            # checkpointer persists the FULL raw conversation (including real
            # tool_calls) server-side under this id, so nothing needs to be
            # tracked or resent from the UI.
            thread_id_state = gr.State(value=lambda: uuid.uuid4().hex)
            msg_box.submit(
                _react_respond,
                inputs=[msg_box, thinking_box, answer_box, thread_id_state],
                outputs=[thinking_box, answer_box, msg_box, thread_id_state],
            )
            submit_btn.click(
                _react_respond,
                inputs=[msg_box, thinking_box, answer_box, thread_id_state],
                outputs=[thinking_box, answer_box, msg_box, thread_id_state],
            )
            clear_btn.click(
                lambda: ([], [], "", uuid.uuid4().hex),
                None,
                [thinking_box, answer_box, msg_box, thread_id_state],
            )

        # with gr.Tab("Rule-based agent (/chat)"):
        #     gr.ChatInterface(
        #         fn=_chat_rule_based,
        #         examples=[
        #             "What is 12 * (7 + 3)?",
        #             "What is the refund policy?",
        #             "What is the current date?",
        #         ],
        #     )
        # with gr.Tab("LCEL chain (/chat/lcel)"):
        #     gr.Markdown(
        #         "Declarative `prompt | model | output_parser` chain -- no tool "
        #         "use, no routing, just the master persona prompt sent straight "
        #         "to the model. See `app.lcel_chain.build_assistant_chain`."
        #     )
        #     gr.ChatInterface(
        #         fn=_chat_lcel,
        #         examples=[
        #             "What is the refund policy?",
        #             "Tell me a fun fact.",
        #         ],
        #     )
        # with gr.Tab("LCEL chain streaming (/chat/lcel/stream)"):
        #     gr.Markdown(
        #         "Same chain as the 'LCEL chain' tab, but built on a real "
        #         "`ChatOllama` Runnable (`prompt | ChatOllama | StrOutputParser`) "
        #         "so the answer streams in token-by-token instead of appearing "
        #         "all at once after a long wait. See "
        #         "`app.lcel_chain.build_assistant_chain_streaming`."
        #     )
        #     gr.ChatInterface(
        #         fn=_chat_lcel_stream,
        #         examples=[
        #             "What is the refund policy?",
        #             "Tell me a fun fact.",
        #         ],
        #     )
        # with gr.Tab("LCEL chain + tools (/chat/lcel/tools)"):
        #     gr.Markdown(
        #         "LCEL chain using LangChain's native `bind_tools()` -- the "
        #         "same tools as the ReAct agent, but composed as a real LCEL "
        #         "`Runnable` instead of a hand-rolled HTTP loop. Requires "
        #         "Ollama running locally. See "
        #         "`app.lcel_chain.build_tool_calling_chain`."
        #     )
        #     gr.ChatInterface(
        #         fn=_chat_lcel_tools,
        #         examples=[
        #             "What is 23 times 17?",
        #             "Check the README and tell me what this project does.",
        #         ],
        #     )
        # with gr.Tab("LCEL chain + tools streaming (/chat/lcel/tools/stream)"):
        #     gr.Markdown(
        #         "Same tool-calling chain as the tab above, but streamed: each "
        #         "round's model output is yielded as it arrives from Ollama, "
        #         "so the FINAL answer (after any tool calls) appears "
        #         "incrementally instead of all at once. See "
        #         "`app.lcel_chain.build_tool_calling_chain_streaming`."
        #     )
        #     gr.ChatInterface(
        #         fn=_chat_lcel_tools_stream,
        #         examples=[
        #             "What is 23 times 17?",
        #             "Check the README and tell me what this project does.",
        #         ],
        #     )
        # with gr.Tab("LCEL chain + tools + memory (/chat/lcel/tools/memory)"):
        #     gr.Markdown(
        #         "Same tool-calling LCEL chain, but wrapped with LangChain's "
        #         "PREBUILT session memory -- `InMemoryChatMessageHistory` + "
        #         "`RunnableWithMessageHistory` -- instead of this agent "
        #         "relying on Gradio's own client-side history. Each browser "
        #         "tab gets its own `session_id` (generated once on load), so "
        #         "the SERVER remembers what was said earlier in `session_id`'s "
        #         "conversation automatically -- try asking it your name, then "
        #         "asking what your name is. See "
        #         "`app.lcel_chain.build_tool_calling_chain_with_memory`."
        #     )
        #     session_id_state = gr.State(value=lambda: str(uuid.uuid4()))
        #     gr.ChatInterface(
        #         fn=_chat_lcel_tools_memory,
        #         additional_inputs=[session_id_state],
        #         # When `additional_inputs` is set, Gradio requires `examples`
        #         # to be a list of ROWS (one list per example), not a flat list
        #         # of strings -- each row is [message, *extra_input_values].
        #         # `session_id_state` is a hidden gr.State (not a visible
        #         # input), so its example value is never actually shown/used;
        #         # None is just a placeholder to satisfy the row shape.
        #         examples=[
        #             ["My name is Ada. Remember that.", None],
        #             ["What is my name?", None],
        #         ],
        #     )
        # with gr.Tab(
        #     "LCEL chain + tools + memory + streaming (/chat/lcel/tools/memory/stream)"
        # ):
        #     gr.Markdown(
        #         "Everything combined: LangChain's native `bind_tools()` tool "
        #         "use, PREBUILT session memory (`InMemoryChatMessageHistory` + "
        #         "`RunnableWithMessageHistory`), AND token-by-token streaming, "
        #         "all on the same chain. See "
        #         "`app.lcel_chain.build_tool_calling_chain_streaming_with_memory`."
        #     )
        #     streaming_memory_session_id_state = gr.State(value=lambda: str(uuid.uuid4()))
        #     gr.ChatInterface(
        #         fn=_chat_lcel_tools_memory_stream,
        #         additional_inputs=[streaming_memory_session_id_state],
        #         examples=[
        #             ["My name is Ada. Remember that.", None],
        #             ["What is my name?", None],
        #         ],
        #     )
        # with gr.Tab("Reasoning model (/chat/reasoning)"):
        #     gr.Markdown(
        #         "Dedicated REASONING-model path (DeepSeek-R1 by default -- "
        #         "see `app.config.OLLAMA_REASONING_MODEL`), separate from the "
        #         "fast/balanced/powerful cost-tier routing used elsewhere in "
        #         "this demo. Unlike every other tab, this model emits a "
        #         "visible chain-of-thought BEFORE its final answer -- shown "
        #         "here in a collapsible '🤔 Show reasoning' section above the "
        #         "answer, via `app.reasoning.generate_with_reasoning`. "
        #         "Requires Ollama running locally with the model pulled "
        #         "(`ollama pull deepseek-r1:14b`). Reasoning models "
        #         "legitimately take much longer than a normal completion -- "
        #         "that's expected, not a hang.\n\n"
        #         "**No tools here, on purpose:** this model claims tool-calling "
        #         "support via Ollama's native mechanism, but testing showed it "
        #         "doesn't reliably use THAT -- it either ignores a provided tool "
        #         "and improvises an answer itself (sometimes wrong despite "
        #         "correct-looking reasoning), or narrates what a tool call WOULD "
        #         "return instead of making a real one. See `app/reasoning.py`'s "
        #         "module docstring for the specific tests. This tab only shows "
        #         "how the model reasons over what it already knows -- for "
        #         "reasoning model that ALSO reliably uses tools, see the "
        #         "'Reasoning + tools' tab (a different, text-based protocol that "
        #         "actually works for this model)."
        #     )
        #     gr.ChatInterface(
        #         fn=_chat_reasoning,
        #         examples=[
        #             "If a train leaves at 3pm going 60mph and another leaves "
        #             "an hour later going 80mph on the same route, when does "
        #             "the second train catch up?",
        #             "Is 1009 a prime number? Show your reasoning.",
        #         ],
        #     )
        # with gr.Tab("Reasoning + tools (/agent/react/reasoning)"):
        #     gr.Markdown(
        #         "The SAME reasoning model (DeepSeek-R1) as the tab above, but "
        #         "WITH reliable tool use -- via "
        #         "`app.reasoning_react_agent.ReasoningReActAgent`. Ollama's "
        #         "native `tools`/`tool_calls` JSON mechanism (used by the "
        #         "'ReAct agent' tab) is NOT reliable for this model, so this "
        #         "uses a different, older protocol instead: the model is "
        #         "instructed to emit plain `Action:` / `Action Input:` text "
        #         "lines (the original ReAct-paper style, predating 'function "
        #         "calling' APIs), which we parse and execute ourselves, "
        #         "feeding the REAL result back as the next turn. This was "
        #         "verified empirically to work reliably (unlike native "
        #         "tool_calls) -- see the module docstring for the exact "
        #         "tests. Shows both the final chain-of-thought AND which "
        #         "tools were actually called.\n\n"
        #         "**⚠️ Known limitation:** reliable for simple, single-tool "
        #         "asks. For multi-step tasks (e.g. 'read this file, then "
        #         "edit it') live testing caught the model sometimes SKIPPING "
        #         "the second tool call and just claiming success anyway -- "
        #         "check the 'Tool calls' list below the answer to see what "
        #         "ACTUALLY ran, don't just trust the final answer text for "
        #         "anything destructive. See the module docstring's "
        #         "'KNOWN REMAINING FAILURE MODE' section for details."
        #     )
        #     gr.ChatInterface(
        #         fn=_chat_reasoning_react,
        #         examples=[
        #             "What is 23 times 17? Use the calculator tool.",
        #             "Count the .py files in the current directory tree.",
        #             "How much memory is this machine using right now?",
        #         ],
        #     )
        # with gr.Tab("Reasoning + tools, LangGraph (/agent/react/reasoning/langgraph)"):
        #     gr.Markdown(
        #         "Same task, model, and text-based `Action:`/`Action Input:` "
        #         "protocol as the 'Reasoning + tools' tab, but orchestrated "
        #         "as an explicit LangGraph `StateGraph` via "
        #         "`app.reasoning_langgraph_agent.ReasoningLangGraphAgent` "
        #         "instead of a plain Python loop. The practical difference: "
        #         "a genuinely MALFORMED tool call (invalid JSON in Action "
        #         "Input) is caught by a conditional graph edge that routes "
        #         "back to the model with the real parse error as feedback -- "
        #         "an explicit self-correction CYCLE, which LangGraph "
        #         "supports natively (unlike a straight-line LCEL chain). "
        #         "See the module docstring for the graph shape."
        #     )
        #     gr.ChatInterface(
        #         fn=_chat_reasoning_langgraph,
        #         examples=[
        #             "What is 23 times 17? Use the calculator tool.",
        #             "How much memory is this machine using right now?",
        #         ],
        #     )
    return demo


if __name__ == "__main__":
    build_app().launch()
