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

from datetime import date

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from app.config import MODEL_CATALOG, OLLAMA_BASE_URL, ModelTier
from app.prompts import ASSISTANT_PERSONA_TEMPLATE, DEFAULT_COMPANY_NAME
from app.providers import LLMProvider, get_default_provider
from app.tools import AVAILABLE_TOOLS

_TOOLS_BY_NAME = {t.name: t for t in AVAILABLE_TOOLS}


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


MAX_TOOL_ROUNDS = 3  # same spirit as ReActAgent.MAX_ITERATIONS -- a hard cap


def _run_tool_calling_loop(llm_with_tools: Runnable, user_request: str) -> str:
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
    """
    messages: list[BaseMessage] = [HumanMessage(content=user_request)]
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
    hand-rolled HTTP loop instead of this LangChain-native path).

    Call ``.invoke({"user_request": "..."})``. Pass a pre-built
    ``llm_with_tools`` Runnable (e.g. a mock, for tests) to avoid constructing
    a real ``ChatOllama`` client.
    """
    if llm_with_tools is None:
        from langchain_ollama import ChatOllama

        resolved_model = model or MODEL_CATALOG["powerful"].ollama_model
        llm_with_tools = ChatOllama(model=resolved_model, base_url=base_url).bind_tools(
            AVAILABLE_TOOLS
        )

    def _invoke(inputs: dict) -> str:
        return _run_tool_calling_loop(llm_with_tools, inputs["user_request"])

    return RunnableLambda(_invoke)
