"""Gradio chat UI for manually testing the agents without curl.

Run with:
    LLM_BACKEND=ollama .venv/bin/python -m app.gradio_app

Then open the printed local URL (usually http://127.0.0.1:7860) in a browser.
Exposes both agents side-by-side as separate tabs so you can directly compare
the deterministic, rule-based agent against the LLM-driven ReAct agent on the
exact same question.
"""
from __future__ import annotations

import gradio as gr

from app.agent import Agent
from app.react_agent import ReActAgent

_rule_based_agent = Agent()
_react_agent = ReActAgent()


def _chat_rule_based(message: str, history: list) -> str:
    result = _rule_based_agent.handle(message)
    meta = (
        f"\n\n---\n*tool: `{result.tool_used or 'none'}` | "
        f"model: `{result.model or 'n/a'}` | tier: `{result.tier or 'n/a'}` | "
        f"cost: ${result.estimated_cost_usd:.5f} | "
        f"reason: {result.routing_reason}*"
    )
    return result.answer + meta


def _chat_react(message: str, history: list) -> str:
    result = _react_agent.handle(message)
    if result.steps:
        steps_md = "\n".join(
            f"{i}. `{s.tool_name}({s.tool_args})` -> {s.observation[:200]}"
            for i, s in enumerate(result.steps, start=1)
        )
    else:
        steps_md = "(no tools called -- answered directly)"
    meta = (
        f"\n\n---\n*model: `{result.model}` | iterations: {result.iterations} | "
        f"hit iteration limit: {result.hit_iteration_limit}*\n\n**Tool calls:**\n{steps_md}"
    )
    return result.answer + meta


def build_app() -> gr.Blocks:
    with gr.Blocks(title="GenAI Agent Demo") as demo:
        gr.Markdown(
            "# GenAI Agent Demo\n"
            "Compare the deterministic **rule-based agent** (regex routing, `/chat`) "
            "against the **LLM-driven ReAct agent** (`/agent/react`) on the same question."
        )
        with gr.Tab("Rule-based agent (/chat)"):
            gr.ChatInterface(
                fn=_chat_rule_based,
                examples=[
                    "What is 12 * (7 + 3)?",
                    "What is the refund policy?",
                    "What is the current date?",
                ],
            )
        with gr.Tab("ReAct agent (/agent/react)"):
            gr.ChatInterface(
                fn=_chat_react,
                examples=[
                    "What is 23 times 17?",
                    "How quickly do you respond to support tickets?",
                    "How much memory is this machine using right now?",
                    "Count the files in the current directory.",
                ],
            )
    return demo


if __name__ == "__main__":
    build_app().launch()
