"""Tools exposed to the agent, defined with LangChain's ``@tool`` decorator.

These illustrate the "agent frameworks" requirement: the agent can decide,
based on the user's request, whether it needs to call a deterministic tool
(e.g. a calculator) instead of hallucinating an answer -- a core idea behind
LangChain/LlamaIndex/AutoGen-style agents.
"""
from __future__ import annotations

import ast
import operator as op

from langchain_core.tools import tool

_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'.

    Use this whenever the user asks for a numeric computation so the answer
    is exact rather than guessed by the LLM.
    """
    try:
        tree = ast.parse(expression, mode="eval").body
        return str(_safe_eval(tree))
    except Exception:
        return "Error: could not evaluate expression"


@tool
def knowledge_base_lookup(topic: str) -> str:
    """Look up a short fact about a known internal topic.

    This simulates a retrieval tool (as you'd build with LlamaIndex over a
    real document/vector store) without requiring external infrastructure.
    """
    mock_kb = {
        "refund policy": "Refunds are processed within 5-7 business days of approval.",
        "sla": "Standard support SLA is a 4 business hour first response time.",
        "pricing": "Pricing tiers are Starter, Growth, and Enterprise, billed monthly.",
    }
    return mock_kb.get(topic.strip().lower(), f"No internal knowledge found for '{topic}'.")


AVAILABLE_TOOLS = [calculator, knowledge_base_lookup]
