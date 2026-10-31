"""Centralized "master" prompt templates, built with LangChain's
``PromptTemplate``.

Why this matters: without this module, every place that talks to an LLM
(``app.agent.Agent``'s LLM fallback path, ``app.react_agent.ReActAgent``'s
system prompt) would hand-build its own raw f-string. That makes it easy for
tone/persona/safety-instructions to drift out of sync between code paths, and
hard to find/change them later ("where do I edit what the assistant is told
it's allowed to do?"). Centralizing them here means:

  - there is exactly ONE place that defines the assistant's persona/voice,
  - templates are parameterized (company name, current date, tool list) via
    LangChain's ``PromptTemplate.format(...)``, instead of duplicated
    f-strings,
  - it's independently testable (see tests/test_prompts.py) without needing
    a real LLM call.

This mirrors the standard LangChain pattern from prompt-templating tutorials:
define a template once with named placeholders, then `.format(**kwargs)` it
wherever it's needed.
"""
from __future__ import annotations

from datetime import date

from langchain_core.prompts import PromptTemplate

DEFAULT_COMPANY_NAME = "Acme GenAI Demo Co."

# ---------------------------------------------------------------------------
# 1. Master persona template, wrapping every PLAIN (no-tool) LLM call made by
#    the rule-based Agent. This is applied only at the final "generate an
#    answer" step -- NOT before model routing, since classify_complexity()
#    needs to see the user's actual words, not this boilerplate, to make a
#    correct cost/latency decision.
# ---------------------------------------------------------------------------
ASSISTANT_PERSONA_TEMPLATE = PromptTemplate.from_template(
    "You are the support assistant for {company_name}. Today's date is "
    "{current_date}. Be concise and factual, and say \"I don't know\" "
    "rather than guessing when you are unsure.\n\n"
    "User request: {user_request}"
)


def render_assistant_prompt(user_request: str, company_name: str = DEFAULT_COMPANY_NAME) -> str:
    """Wrap a raw user request in the master persona template."""
    return ASSISTANT_PERSONA_TEMPLATE.format(
        company_name=company_name,
        current_date=date.today().isoformat(),
        user_request=user_request,
    )


# ---------------------------------------------------------------------------
# 2. Master ReAct system prompt, parameterized by the ACTUAL tool names
#    available at runtime. Deriving {tool_names} from the real tool registry
#    (rather than hand-listing them in the template text) means the prompt
#    can never silently drift out of sync with what tools really exist.
# ---------------------------------------------------------------------------
REACT_SYSTEM_TEMPLATE = PromptTemplate.from_template(
    "You are the support assistant for {company_name}. Today's date is "
    "{current_date}. You have access to these tools: {tool_names}. Use a "
    "tool whenever it would give a more accurate or verifiable answer than "
    "guessing (e.g. arithmetic, internal company facts, OS/system "
    "information, or anything about the actual codebase/files on disk). If "
    "no tool is needed, answer directly. Do not call a tool more than once "
    "with the same arguments. IMPORTANT: when you decide to use a tool, you "
    "MUST call it through the tool-calling mechanism provided to you -- "
    "never write out a JSON object describing the tool call as your answer "
    "text, since that will not actually run it. ALSO IMPORTANT: if "
    "answering correctly requires verifying something (e.g. whether a file, "
    "code pattern, or setting exists), you MUST actually call the tool and "
    "use its real output -- do NOT describe, suggest, or explain which "
    "tool(s) the user (or you) could call instead of calling them yourself. "
    "A description of your plan is never an acceptable substitute for "
    "actually executing it."
)


def render_react_system_prompt(
    tool_names: list[str], company_name: str = DEFAULT_COMPANY_NAME
) -> str:
    return REACT_SYSTEM_TEMPLATE.format(
        company_name=company_name,
        current_date=date.today().isoformat(),
        tool_names=", ".join(tool_names),
    )
