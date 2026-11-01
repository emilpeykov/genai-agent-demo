"""A LangChain Expression Language (LCEL) chain -- an alternative,
declarative way to compose "prompt -> model -> parse the output" that's
worth knowing alongside the explicit, imperative control flow used
elsewhere in this project (``app.agent.Agent.handle()``,
``app.react_agent.ReActAgent.handle()``).

What LCEL actually is: every piece (a ``PromptTemplate``, a model, an output
parser, even a plain function wrapped in ``RunnableLambda``) implements the
same ``Runnable`` interface (``.invoke()``, ``.batch()``, ``.stream()``, and
async equivalents). The ``|`` operator just chains them so each one's output
becomes the next one's input:

    chain = prompt | model | output_parser
    chain.invoke({"user_request": "What is the refund policy?"})

Why THIS project doesn't use LCEL for its main agent loops: ``Agent.handle()``
and ``ReActAgent.handle()`` return rich telemetry alongside the answer text
(model name, tier, cost, retries, whether a fallback happened) -- information
a plain LCEL chain's final output doesn't naturally carry through, since a
chain's output is typically just the parsed text. Our custom
``app.resilience.call_with_resilience`` also implements tier-FALLBACK (not
just same-tier retries), which is more involved than LCEL's built-in
``.with_retry()``. Rather than compromise that telemetry, this module keeps
LCEL as a clean, separate demonstration.

This mirrors ``ASSISTANT_PERSONA_TEMPLATE`` from ``app.prompts`` (the exact
same master template used by ``Agent``'s plain-LLM fallback path) so it's a
real, working alternative construction of the same prompt -> answer flow,
not a toy example.

Does LCEL support tools? Yes -- see ``build_tool_calling_chain`` below, which
uses LangChain's standard ``some_chat_model.bind_tools(AVAILABLE_TOOLS)``
mechanism. It needs a real LangChain ``BaseChatModel`` (``ChatOllama``) to do
this, which is why ``build_assistant_chain`` above -- built on our own
text-in/text-out ``LLMProvider`` -- has no tools wired in: that's an
architectural choice (to keep ``Agent``'s cost/tier telemetry working), not a
limitation of LCEL itself.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import date

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnableGenerator, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.config import MODEL_CATALOG, OLLAMA_BASE_URL, ModelTier
from app.prompts import ASSISTANT_PERSONA_TEMPLATE, DEFAULT_COMPANY_NAME, render_react_system_prompt
from app.providers import LLMProvider, get_default_provider
from app.tools import AVAILABLE_TOOLS

_TOOLS_BY_NAME = {t.name: t for t in AVAILABLE_TOOLS}

# Same master system prompt ReActAgent uses (app.prompts.REACT_SYSTEM_TEMPLATE),
# rendered once with the REAL tool names/descriptions available here. Without
# this, the model was never told HOW to call tools correctly, nor that
# conversation memory is automatic -- in practice this caused a real bug: a
# lightweight model, asked to "remember" a fact, would try to literally call
# 'write_file' to persist it (and sometimes emit the tool call as raw JSON
# text instead of a real structured tool call, since nothing told it not to).
# ReActAgent's system prompt already explicitly rules both mistakes out, so
# reusing it here (rather than leaving these chains with NO system message at
# all) fixes the same class of bug in this code path too.
_SYSTEM_PROMPT = render_react_system_prompt(tool_names=[t.name for t in AVAILABLE_TOOLS])

# Per-session chat history store, keyed by an arbitrary caller-supplied
# ``session_id``. Uses LangChain's own PREBUILT ``InMemoryChatMessageHistory``
# (langchain_core.chat_history) rather than hand-rolling a dict-of-lists
# ourselves -- same "prefer the maintained library tool" principle used for
# the file/Wikipedia tools in app.tools. This is process-memory only (lost on
# restart, not shared across multiple server workers/processes) -- fine for a
# local demo; a real deployment would back this with Redis/a database via a
# custom ``BaseChatMessageHistory`` subclass instead.
_SESSION_STORE: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return (creating on first use) the message history for ``session_id``."""
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return _SESSION_STORE[session_id]


def _prompt_value_to_text(prompt_value) -> str:
    """``ASSISTANT_PERSONA_TEMPLATE`` (a ``PromptTemplate``) outputs a
    ``PromptValue`` object when used as a Runnable, not a plain string --
    this converts it to the plain text our ``LLMProvider.generate()``
    expects, the same way a real chat-model Runnable would consume it
    internally.
    """
    return prompt_value.to_string()


def build_prompt_model_parser_chain(
    template: BasePromptTemplate,
    provider: LLMProvider | None = None,
    tier: ModelTier = "fast",
    output_parser: BaseOutputParser | None = None,
    retries: int = 2,
) -> Runnable:
    """Generic ``prompt | model | output_parser`` chain builder.

    Takes ANY ``PromptTemplate`` (already ``.partial()``-ed with whatever
    fixed values it needs), formats it with the input dict passed to
    ``.invoke()``, sends the rendered text to ``provider.generate()`` at the
    given ``tier``, then runs the result through ``output_parser`` (defaults
    to ``StrOutputParser``, a no-op pass-through for plain text).

    This is the reusable building block behind ``build_assistant_chain``;
    use it directly for other templates (e.g. ``REACT_SYSTEM_TEMPLATE``) or
    custom ones defined ad hoc, whenever you want the "format a template ->
    call the model -> parse the output" shape without hand-rolling it again.
    """
    active_provider = provider or get_default_provider()
    parser = output_parser or StrOutputParser()

    def _call_provider(prompt_text: str) -> str:
        return active_provider.generate(prompt_text, tier).text

    chain = (
        template
        | RunnableLambda(_prompt_value_to_text)
        | RunnableLambda(_call_provider)
        | parser
    )
    # LCEL's built-in retry -- mirrors app.resilience's same-tier retry loop,
    # though (unlike call_with_resilience) it has no concept of falling back
    # to a cheaper MODEL TIER; it only retries the SAME chain again.
    return chain.with_retry(stop_after_attempt=retries)


def build_assistant_chain(
    provider: LLMProvider | None = None,
    tier: ModelTier = "fast",
) -> Runnable:
    """Build an LCEL chain equivalent to the plain-LLM path in
    ``Agent.handle()``: render the master persona prompt, call the model,
    and parse the output -- expressed as ``prompt | model | output_parser``
    instead of three separate imperative steps.

    Demonstrates:
      - ``PromptTemplate`` used directly as the first link in a chain (it's
        already a ``Runnable`` -- no wrapping needed).
      - ``RunnableLambda`` for wrapping plain Python functions (here, our
        existing ``LLMProvider`` abstraction) so arbitrary code can be a
        link in the chain too, not just LangChain-native model classes.
      - ``StrOutputParser`` for normalizing the final output to a plain
        string (a no-op here since our provider already returns text, but
        it's the standard last link when a model instead returns a
        structured message object).
      - ``.with_retry()``: LCEL's built-in retry primitive, applied to the
        whole chain -- the LCEL equivalent of (a simplified version of)
        ``app.resilience.call_with_resilience``'s same-tier retry loop.

    Returns a ``Runnable`` -- call ``.invoke({"user_request": "..."})`` on
    it. ``company_name``/``current_date`` are baked in ahead of time via
    ``PromptTemplate.partial()`` (another LCEL/LangChain primitive: filling
    in some variables now so callers of the final chain only need to supply
    the ones that actually vary per-call).
    """
    partial_prompt = ASSISTANT_PERSONA_TEMPLATE.partial(
        company_name=DEFAULT_COMPANY_NAME,
        current_date=date.today().isoformat(),
    )
    return build_prompt_model_parser_chain(partial_prompt, provider=provider, tier=tier)


def build_assistant_chain_streaming(
    model: str | None = None,
    base_url: str = OLLAMA_BASE_URL,
    tier: ModelTier = "fast",
) -> Runnable:
    """Same prompt/persona as ``build_assistant_chain``, but built as
    ``prompt | ChatOllama | StrOutputParser()`` -- a chain of REAL LangChain
    ``Runnable``s -- instead of ``prompt | RunnableLambda(provider.generate)``.

    Why this is a separate chain rather than just "making build_assistant_chain
    stream": ``RunnableLambda`` wraps a plain function that returns ONE value
    when it's done -- there's no way for LCEL to get partial output out of it
    early, no matter how it's called. True incremental (token-by-token)
    streaming only works through the chain's ``.stream()``/``.astream()``
    when EVERY link is a real streaming-capable ``Runnable``: ``ChatOllama``
    implements ``_stream()`` natively, and ``StrOutputParser`` forwards each
    incoming ``AIMessageChunk`` as the growing string of ``.content`` so far.
    This is exactly the same "use our own LLMProvider for telemetry vs. a
    real BaseChatModel for LangChain-native features" tradeoff already made
    for ``build_tool_calling_chain`` -- streaming, like tool-calling, needs a
    real chat-model Runnable.

    Call ``.stream({"user_request": "..."})`` and iterate the result for
    incremental chunks, or ``.invoke(...)``/``.ainvoke(...)`` as usual for
    the whole answer at once (LCEL chains support both from the same
    definition -- no separate "streaming version" of the chain is needed
    beyond swapping which link talks to the model).
    """
    from langchain_ollama import ChatOllama

    resolved_model = model or MODEL_CATALOG[tier].ollama_model
    llm = ChatOllama(model=resolved_model, base_url=base_url)
    partial_prompt = ASSISTANT_PERSONA_TEMPLATE.partial(
        company_name=DEFAULT_COMPANY_NAME,
        current_date=date.today().isoformat(),
    )
    return partial_prompt | llm | StrOutputParser()


MAX_TOOL_ROUNDS = 3  # same spirit as ReActAgent.MAX_ITERATIONS -- a hard cap


def _run_tool_calling_loop(
    llm_with_tools: Runnable,
    user_request: str,
    chat_history: list[BaseMessage] | None = None,
) -> str:
    """Drive a bounded tool-calling loop against an LLM already bound with
    tools (``llm_with_tools = some_chat_model.bind_tools(AVAILABLE_TOOLS)``).

    This is the part LCEL's static ``|`` composition can't express directly:
    "call the model, and if it asked for a tool, call that tool and go back
    to the model again" is a LOOP with a data-dependent exit condition, not a
    fixed pipeline. The standard way to fit this into LCEL is exactly what's
    done here -- wrap the looping logic in a plain function/``RunnableLambda``
    that internally calls one or more *real* Runnables (``llm_with_tools``)
    as many times as needed. ``ReActAgent`` solves the identical problem by
    hand-rolling the same loop directly against Ollama's HTTP API instead of
    going through a LangChain chat-model Runnable.

    ``chat_history`` (optional) is a list of prior turns' ``BaseMessage``s --
    e.g. supplied by ``RunnableWithMessageHistory`` in
    ``build_tool_calling_chain_with_memory`` -- prepended before the new
    ``user_request`` so the model has real conversational memory. Omit it
    (the default) for the original stateless, single-turn behavior.
    """
    messages: list[BaseMessage] = (
        [SystemMessage(content=_SYSTEM_PROMPT)]
        + list(chat_history or [])
        + [HumanMessage(content=user_request)]
    )
    last_ai_message: AIMessage | None = None

    for _ in range(MAX_TOOL_ROUNDS):
        ai_message = llm_with_tools.invoke(messages)
        last_ai_message = ai_message
        messages.append(ai_message)

        if not ai_message.tool_calls:
            return ai_message.content

        for call in ai_message.tool_calls:
            tool = _TOOLS_BY_NAME.get(call["name"])
            if tool is None:
                result = f"Error: unknown tool '{call['name']}'"
            else:
                try:
                    result = tool.invoke(call["args"])
                except Exception as exc:  # defensive: never let a bad tool call crash the chain
                    result = f"Error running tool '{call['name']}': {exc}"
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    # Hit the round cap -- return the model's own last message content rather
    # than raising, mirroring ReActAgent's "hit_iteration_limit" path.
    return str(last_ai_message.content) if last_ai_message is not None else ""


def build_tool_calling_chain(
    llm_with_tools: Runnable | None = None,
    model: str | None = None,
    base_url: str = OLLAMA_BASE_URL,
) -> Runnable:
    """A genuine tool-using LCEL chain -- answers the question "does LCEL
    support tools?" with working code instead of just "yes, via
    ``.bind_tools()``".

    Unlike ``build_assistant_chain``/``build_prompt_model_parser_chain``
    (which call our own text-in/text-out ``LLMProvider`` abstraction, chosen
    there so ``Agent``'s cost/latency/tier telemetry keeps working), this
    builds on a REAL LangChain ``BaseChatModel`` (``ChatOllama``) so
    ``.bind_tools(AVAILABLE_TOOLS)`` -- LCEL's native, LangChain-standard way
    to give a chain access to tools -- can be used directly, the same
    mechanism used by LangChain's own prebuilt agents.

    Requires Ollama running locally with a tool-calling-capable model (see
    ``ReActAgent``, which uses the same tools against the same backend via a
    hand-rolled HTTP loop instead of this LangChain-native path, and defaults
    to the SAME ``"balanced"`` tier model for the same reason -- see below).

    Call ``.invoke({"user_request": "..."})``. Pass a pre-built
    ``llm_with_tools`` Runnable (e.g. a mock, for tests) to avoid constructing
    a real ``ChatOllama`` client.
    """
    if llm_with_tools is None:
        from langchain_ollama import ChatOllama

        # "powerful" (devstral:24b): llama3.1:8b ("balanced") was tried here
        # first for its lighter memory footprint, but it turned out to be
        # UNRELIABLE at structured tool-calling via LangChain's bind_tools()
        # -- observed emitting malformed tool-call-shaped text directly as
        # message content (e.g. '{"name": "None", "parameters": {}}')
        # instead of a real tool call, non-deterministically, even for
        # simple turns that needed no tool at all. devstral:24b is larger
        # (14GB) and will compete for memory with other loaded models, but
        # is noticeably more consistent here -- correctness beats raw
        # memory footprint for this demo. If memory pressure becomes a
        # problem again, consider running `ollama stop <model>` on unused
        # models, or override via the ``model=`` parameter.
        resolved_model = model or MODEL_CATALOG["powerful"].ollama_model
        llm_with_tools = ChatOllama(model=resolved_model, base_url=base_url).bind_tools(
            AVAILABLE_TOOLS
        )

    def _invoke(inputs: dict) -> str:
        return _run_tool_calling_loop(
            llm_with_tools, inputs["user_request"], inputs.get("chat_history")
        )

    return RunnableLambda(_invoke)


def _stream_tool_calling_loop(
    llm_with_tools: Runnable,
    user_request: str,
    chat_history: list[BaseMessage] | None = None,
) -> Iterator[str]:
    """Streaming counterpart to ``_run_tool_calling_loop``: yields the
    model's answer text incrementally (token-by-token, as the chunks arrive
    from Ollama) instead of returning one complete string at the end.

    Each round calls ``llm_with_tools.stream(messages)`` instead of
    ``.invoke(messages)``, accumulating the ``AIMessageChunk``s with ``+``
    (LangChain message chunks implement ``__add__`` precisely so a stream can
    be reassembled into the same complete message ``.invoke()`` would have
    returned, including ``.tool_calls``). Any text content chunks are yielded
    to the caller AS THEY ARRIVE.

    In practice, a round where the model decides to call a tool emits little
    or no natural-language ``content`` (just tool-call tokens, which live in
    ``.tool_call_chunks``, not ``.content``) -- so yielding ``chunk.content``
    unconditionally on every round is safe: tool-deciding rounds simply yield
    nothing (or a few stray words some models emit), and only the FINAL round
    (the one with no tool calls) produces the real streamed answer text.
    """
    messages: list[BaseMessage] = (
        [SystemMessage(content=_SYSTEM_PROMPT)]
        + list(chat_history or [])
        + [HumanMessage(content=user_request)]
    )

    for _ in range(MAX_TOOL_ROUNDS):
        full: AIMessage | None = None
        for chunk in llm_with_tools.stream(messages):
            full = chunk if full is None else full + chunk
            if chunk.content:
                yield chunk.content

        if full is None:  # defensive: model produced an empty stream
            return
        messages.append(full)

        if not full.tool_calls:
            return  # that was the final answer -- already streamed above

        for call in full.tool_calls:
            tool = _TOOLS_BY_NAME.get(call["name"])
            if tool is None:
                result = f"Error: unknown tool '{call['name']}'"
            else:
                try:
                    result = tool.invoke(call["args"])
                except Exception as exc:  # defensive: never let a bad tool call crash the chain
                    result = f"Error running tool '{call['name']}': {exc}"
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    # Hit the round cap without a final text-only round -- nothing further to
    # stream (mirrors _run_tool_calling_loop's "last_ai_message" fallback,
    # except there is no extra text to emit since it would already have been
    # yielded above if the model produced any).


def build_tool_calling_chain_streaming(
    llm_with_tools: Runnable | None = None,
    model: str | None = None,
    base_url: str = OLLAMA_BASE_URL,
) -> Runnable:
    """Streaming counterpart to ``build_tool_calling_chain``.

    Wraps ``_stream_tool_calling_loop`` in a ``RunnableGenerator``
    (``langchain_core.runnables``) rather than ``RunnableLambda``, since a
    ``RunnableLambda`` can only return one finished value -- a
    ``RunnableGenerator`` is LCEL's primitive for a link that yields a
    sequence of outputs, which is what makes ``.stream(...)`` on the
    returned chain actually emit chunks incrementally instead of blocking
    until the whole answer is ready. ``.invoke(...)`` still works exactly as
    before (it consumes the full generator and concatenates the string
    chunks), so this is a drop-in replacement wherever ``.invoke()`` was
    used, with ``.stream()`` now also available.
    """
    if llm_with_tools is None:
        from langchain_ollama import ChatOllama

        # Same "powerful" tier choice as build_tool_calling_chain above -- see
        # the comment there for why (more reliable structured tool-calling).
        resolved_model = model or MODEL_CATALOG["powerful"].ollama_model
        llm_with_tools = ChatOllama(model=resolved_model, base_url=base_url).bind_tools(
            AVAILABLE_TOOLS
        )

    def _transform(input_iter: Iterator[dict]) -> Iterator[str]:
        for inputs in input_iter:
            yield from _stream_tool_calling_loop(
                llm_with_tools, inputs["user_request"], inputs.get("chat_history")
            )

    return RunnableGenerator(_transform)


def build_tool_calling_chain_with_memory(
    llm_with_tools: Runnable | None = None,
    model: str | None = None,
    base_url: str = OLLAMA_BASE_URL,
) -> RunnableWithMessageHistory:
    """Same tool-calling chain as ``build_tool_calling_chain``, but wrapped
    with LangChain's PREBUILT session-memory machinery --
    ``InMemoryChatMessageHistory`` + ``RunnableWithMessageHistory``
    (``langchain_core.chat_history``/``langchain_core.runnables.history``) --
    instead of requiring the caller to manually collect and resend prior
    turns themselves (the pattern ``ReActAgent``/``/agent/react`` use via
    their ``history`` parameter).

    Call ``.invoke({"user_request": "..."}, config={"configurable": {"session_id": "..."}})``:
    everything said in that ``session_id`` is automatically looked up before
    the call and recorded again after it, so a second call with the SAME
    ``session_id`` has real conversational memory -- no caller-managed
    ``history`` list required. Different ``session_id``s are fully isolated
    from one another.
    """
    base_chain = build_tool_calling_chain(
        llm_with_tools=llm_with_tools, model=model, base_url=base_url
    )
    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="user_request",
        history_messages_key="chat_history",
    )


def build_tool_calling_chain_streaming_with_memory(
    llm_with_tools: Runnable | None = None,
    model: str | None = None,
    base_url: str = OLLAMA_BASE_URL,
) -> RunnableWithMessageHistory:
    """Combines ``build_tool_calling_chain_streaming`` with the same
    ``RunnableWithMessageHistory`` session memory used by
    ``build_tool_calling_chain_with_memory`` -- i.e. streamed output AND
    server-side conversational memory, keyed by ``session_id``, at the same
    time. ``RunnableWithMessageHistory`` is itself just a wrapper that reads
    history before, and records it after, any underlying ``Runnable`` -- it
    works identically whether that Runnable is a ``RunnableLambda`` (one
    final value) or a ``RunnableGenerator`` (a stream of chunks), so no
    other changes are needed to get both features together.

    Call ``.stream({"user_request": "..."}, config={"configurable": {"session_id": "..."}})``.
    """
    base_chain = build_tool_calling_chain_streaming(
        llm_with_tools=llm_with_tools, model=model, base_url=base_url
    )
    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="user_request",
        history_messages_key="chat_history",
    )

