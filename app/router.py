"""Model routing: pick the cheapest model tier that can likely handle a given
request, based on prompt complexity signals.

This directly demonstrates the "model routing, cost/latency tradeoffs"
requirement: instead of always calling the most powerful (and expensive)
model, we classify each request and send it to the most cost-effective tier
that should still produce a good answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import ModelTier

# Keywords that hint the task needs deeper reasoning (math/logic/code/multi-step).
COMPLEX_SIGNALS = re.compile(
    r"\b(prove|derive|optimi[sz]e|architecture|design a|step[- ]by[- ]step|"
    r"multi-?step|algorithm|trade-?off|root cause|debug|refactor|"
    r"compare and contrast|strategy|plan)\b",
    re.IGNORECASE,
)

MODERATE_SIGNALS = re.compile(
    r"\b(summari[sz]e|explain|analy[sz]e|classify|extract|translate|rewrite)\b",
    re.IGNORECASE,
)


@dataclass
class RoutingDecision:
    tier: ModelTier
    reason: str


def classify_complexity(prompt: str) -> RoutingDecision:
    """Heuristic complexity classifier.

    In a real system this first-pass classifier is often itself a cheap LLM
    call or a lightweight trained classifier; here we use transparent,
    testable heuristics (length + keyword signals) to keep the demo fast,
    deterministic, and free to run.
    """
    word_count = len(prompt.split())

    if COMPLEX_SIGNALS.search(prompt) or word_count > 120:
        return RoutingDecision(
            tier="powerful",
            reason="Prompt contains multi-step/reasoning signals or is long.",
        )

    if MODERATE_SIGNALS.search(prompt) or word_count > 30:
        return RoutingDecision(
            tier="balanced",
            reason="Prompt requires moderate synthesis (summarize/explain/etc.).",
        )

    return RoutingDecision(
        tier="fast",
        reason="Short, simple prompt; cheapest tier should suffice.",
    )
