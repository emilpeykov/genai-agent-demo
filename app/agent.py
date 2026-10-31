"""Agent orchestration layer.

Wires together:
  - tool-use decisions (LangChain tools) for deterministic sub-tasks,
  - model routing (cheapest viable tier) for everything else,
  - resilience (retries + fallback) around the actual LLM call.

This is intentionally a lightweight, explicit control loop rather than a
black-box ``AgentExecutor`` so each decision (tool vs. LLM, which tier, what
happened on failure) is inspectable -- useful both for the demo and for real
production debugging.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.prompts import render_assistant_prompt
from app.providers import LLMProvider, get_default_provider
from app.resilience import AllTiersFailedError, call_with_resilience
from app.router import classify_complexity
from app.tools import AVAILABLE_TOOLS

_MATH_PATTERN = re.compile(r"^\s*[-+]?[\d\s().+\-*/]+\s*$")
_KB_TRIGGER = re.compile(r"\b(refund policy|sla|pricing)\b", re.IGNORECASE)

# Maps a user-intent phrase to the exact, fixed OS command we'll run on their
# behalf. Note we never pass the user's raw text as the command itself --
# we map *intent* to one specific, pre-approved command. This keeps the
# rule-based agent's own behavior safe/deterministic even though
# run_shell_command itself now runs arbitrary shell commands with no
# allowlist (the LLM-driven ReActAgent is the one that relies on that full
# freedom, deciding the exact command text itself).
_SHELL_INTENTS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(what time|current time|current date|what('s| is) (the )?date|today'?s date)\b",
            re.IGNORECASE,
        ),
        "date",
    ),
    (re.compile(r"\b(current (directory|folder)|working directory|pwd)\b", re.IGNORECASE), "pwd"),
    (re.compile(r"\b(disk (space|usage)|free space)\b", re.IGNORECASE), "df -h"),
    (re.compile(r"\b(who am i|current user)\b", re.IGNORECASE), "whoami"),
    (re.compile(r"\b(system info|os version|operating system)\b", re.IGNORECASE), "uname -a"),
]

_TOOLS_BY_NAME = {t.name: t for t in AVAILABLE_TOOLS}


@dataclass
class AgentResult:
    answer: str
    tool_used: str | None
    model: str | None
    tier: str | None
    latency_ms: int
    estimated_cost_usd: float
    retries: int
    degraded: bool
    routing_reason: str


class Agent:
    def __init__(self, provider: LLMProvider | None = None) -> None:
        self._provider = provider or get_default_provider()

    def handle(self, prompt: str) -> AgentResult:
        # 1. Deterministic tool-use short-circuit: cheaper, faster, and more
        #    reliable than asking an LLM to do arithmetic or recall static facts.
        if _MATH_PATTERN.match(prompt):
            result = _TOOLS_BY_NAME["calculator"].invoke({"expression": prompt})
            return AgentResult(
                answer=result,
                tool_used="calculator",
                model=None,
                tier=None,
                latency_ms=0,
                estimated_cost_usd=0.0,
                retries=0,
                degraded=False,
                routing_reason="Matched arithmetic expression; routed to calculator tool.",
            )

        kb_match = _KB_TRIGGER.search(prompt)
        if kb_match:
            result = _TOOLS_BY_NAME["knowledge_base_lookup"].invoke({"topic": kb_match.group(0)})
            return AgentResult(
                answer=result,
                tool_used="knowledge_base_lookup",
                model=None,
                tier=None,
                latency_ms=0,
                estimated_cost_usd=0.0,
                retries=0,
                degraded=False,
                routing_reason="Matched internal knowledge-base topic; routed to KB tool.",
            )

        # 2. Known OS-info intents: map to one specific, pre-approved command.
        for pattern, command in _SHELL_INTENTS:
            if pattern.search(prompt):
                result = _TOOLS_BY_NAME["run_shell_command"].invoke({"command": command})
                return AgentResult(
                    answer=result,
                    tool_used="run_shell_command",
                    model=None,
                    tier=None,
                    latency_ms=0,
                    estimated_cost_usd=0.0,
                    retries=0,
                    degraded=False,
                    routing_reason=f"Matched OS-info intent; ran allowed command '{command}'.",
                )

        # 3. Otherwise: route to the cheapest viable model tier, with
        #    resilience (retry + fallback) around the call.
        #
        #    IMPORTANT: routing is decided from the RAW prompt (classify_
        #    complexity needs to see the user's actual words, not template
        #    boilerplate) -- the master persona template is only applied to
        #    the text that's actually sent to the model for generation.
        decision = classify_complexity(prompt)
        rendered_prompt = render_assistant_prompt(prompt)
        try:
            result = call_with_resilience(self._provider, rendered_prompt, decision.tier)
        except AllTiersFailedError as exc:
            return AgentResult(
                answer=f"Sorry, I couldn't process this request right now ({exc}).",
                tool_used=None,
                model=None,
                tier=decision.tier,
                latency_ms=0,
                estimated_cost_usd=0.0,
                retries=0,
                degraded=True,
                routing_reason=decision.reason,
            )

        return AgentResult(
            answer=result.response.text,
            tool_used=None,
            model=result.response.model,
            tier=result.response.tier,
            latency_ms=result.response.latency_ms,
            estimated_cost_usd=result.response.estimated_cost_usd,
            retries=result.attempts - 1,
            degraded=result.degraded,
            routing_reason=decision.reason,
        )
