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
"""
from __future__ import annotations

from datetime import date

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from app.config import ModelTier
from app.prompts import ASSISTANT_PERSONA_TEMPLATE, DEFAULT_COMPANY_NAME
from app.providers import LLMProvider, get_default_provider


def _prompt_value_to_text(prompt_value) -> str:
    """``ASSISTANT_PERSONA_TEMPLATE`` (a ``PromptTemplate``) outputs a
    ``PromptValue`` object when used as a Runnable, not a plain string --
    this converts it to the plain text our ``LLMProvider.generate()``
    expects, the same way a real chat-model Runnable would consume it
    internally.
    """
    return prompt_value.to_string()


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
    active_provider = provider or get_default_provider()

    def _call_provider(prompt_text: str) -> str:
        return active_provider.generate(prompt_text, tier).text

    partial_prompt = ASSISTANT_PERSONA_TEMPLATE.partial(
        company_name=DEFAULT_COMPANY_NAME,
        current_date=date.today().isoformat(),
    )

    chain = (
        partial_prompt
        | RunnableLambda(_prompt_value_to_text)
        | RunnableLambda(_call_provider)
        | StrOutputParser()
    )
    # LCEL's built-in retry -- mirrors app.resilience's same-tier retry loop,
    # though (unlike call_with_resilience) it has no concept of falling back
    # to a cheaper MODEL TIER; it only retries the SAME chain again.
    return chain.with_retry(stop_after_attempt=2)
