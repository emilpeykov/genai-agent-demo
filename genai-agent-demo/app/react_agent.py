"""A genuine LLM-driven ReAct (Reason + Act) tool-calling agent.

Unlike ``app.agent.Agent`` (which decides tool-vs-LLM via our own regex
rules), this version lets the LLM itself decide, at each turn, whether to
call a tool and which one -- the "real" agent pattern discussed alongside
frameworks like LangChain's AgentExecutor, AutoGen, etc.

Mechanically this is the ReAct loop:
    1. Send the model the conversation so far + the tool schemas available.
    2. The model replies with EITHER a tool call OR a final answer.
    3. If a tool call: execute it, append the result to the conversation,
       go back to step 1 (the model now sees one more fact than before).
    4. If a final answer: stop and return it.

Uses Ollama's ``/api/chat`` endpoint directly (not the LangChain wrapper)
so the request/response mechanics stay fully visible for learning purposes.
Requires a tool-calling-capable local model (llama3.1 and qwen2.5 both
support this).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx

from app.config import MODEL_CATALOG, OLLAMA_BASE_URL
from app.prompts import DEFAULT_COMPANY_NAME, render_react_system_prompt
from app.tools import AVAILABLE_TOOLS

MAX_ITERATIONS = 5  # hard cap so a confused model can't loop forever
# Larger models (e.g. the 24B "powerful" tier model used by default below)
# are noticeably slower per call on CPU/consumer GPUs than the smaller 8B
# "balanced" model, so this is generous enough to avoid spurious timeouts.
REACT_TIMEOUT_SECONDS = 180

_TOOLS_BY_NAME = {t.name: t for t in AVAILABLE_TOOLS}


def _extract_text_tool_call(content: str) -> dict | None:
    """Best-effort fallback for models that narrate a tool call as plain-text
    JSON in the message content instead of using Ollama's real structured
    tool-calling protocol -- e.g. writing out
    ``{"name": "run_shell_command", "parameters": {"cmd": "..."}}`` as if
    that WERE its final answer, rather than actually invoking the tool.

    Without this fallback such a message is indistinguishable from a genuine
    final answer (``message["tool_calls"]`` is empty either way), so the loop
    would stop immediately and the tool would silently never run -- exactly
    the "the LLM suggests a tool but never executes it" symptom this guards
    against. Smaller local models (e.g. llama3.1:8b) do this occasionally
    even though the system prompt tells them not to.

    Returns ``{"name": ..., "arguments": {...}}`` if a JSON object with a
    recognized tool ``name`` is found anywhere in ``content``, else ``None``.
    """
    if not content:
        return None
    text = content.strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if name not in _TOOLS_BY_NAME:
            continue
        args = obj.get("arguments")
        if not isinstance(args, dict):
            args = obj.get("parameters")
        if not isinstance(args, dict):
            args = {}
        return {"name": name, "arguments": args}
    return None


def _mentions_tool_without_calling(content: str) -> bool:
    """Heuristic: True if the text name-drops one of our real tools (e.g.
    "you could use run_shell_command to check...") even though no tool was
    actually called this turn -- i.e. the model described a plan instead of
    executing it. Used to nudge the model into actually calling the tool
    rather than just talking about doing so.
    """
    if not content:
        return False
    lowered = content.lower()
    return any(name in lowered for name in _TOOLS_BY_NAME)


def _build_tool_schemas() -> list[dict]:
    """Convert our LangChain @tool-decorated functions into the JSON schema
    format Ollama's (and OpenAI's) tool-calling APIs expect."""
    schemas = []
    for t in AVAILABLE_TOOLS:
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": {
                        "type": "object",
                        "properties": t.args,
                        "required": list(t.args.keys()),
                    },
                },
            }
        )
    return schemas


@dataclass
class ReActStep:
    tool_name: str
    tool_args: dict
    observation: str


@dataclass
class ReActResult:
    answer: str
    steps: list[ReActStep] = field(default_factory=list)
    iterations: int = 0
    model: str = ""
    hit_iteration_limit: bool = False


class ReActAgent:
    """LLM-driven tool-calling agent using a local Ollama model."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str = OLLAMA_BASE_URL,
        company_name: str = DEFAULT_COMPANY_NAME,
    ) -> None:
        # Default to the "powerful" tier (devstral:24b locally) rather than
        # "balanced" (llama3.1:8b) -- the 8B model was unreliable at basic
        # reasoning/tool-use (e.g. couldn't correctly answer questions about
        # its own codebase, and sometimes narrated tool calls as plain text
        # instead of actually invoking them). The larger model is slower per
        # call but far more reliable, which matters more for a demo meant to
        # showcase agentic tool-use. Override via the `model=` constructor
        # arg (or OLLAMA_POWERFUL_MODEL/OLLAMA_BALANCED_MODEL env vars) if
        # you want to trade accuracy for speed.
        self._model = model or MODEL_CATALOG["powerful"].ollama_model
        self._base_url = base_url.rstrip("/")
        self._tool_schemas = _build_tool_schemas()
        # Rendered once at construction time from the "master" template in
        # app.prompts, parameterized by the REAL tool names this agent has
        # access to -- so the prompt text can never drift out of sync with
        # what tools actually exist.
        self._system_prompt = render_react_system_prompt(
            tool_names=[t.name for t in AVAILABLE_TOOLS],
            company_name=company_name,
        )

    def _call_model(self, messages: list[dict]) -> dict:
        response = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "tools": self._tool_schemas,
                "stream": False,
            },
            timeout=REACT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    def handle(self, prompt: str, history: list[dict] | None = None) -> ReActResult:
        """Run the ReAct loop for ``prompt``.

        ``history`` is an optional list of prior ``{"role": ..., "content": ...}``
        turns (roles "user"/"assistant") from earlier in the conversation --
        without it, every call is completely independent and the model has no
        memory of anything said before (e.g. it won't know what "it" refers
        to in a follow-up question). Tool-call/observation turns from earlier
        exchanges are intentionally NOT replayed here (they'd bloat the
        context very fast); only the user-visible text of each turn is kept.
        """
        messages = [{"role": "system", "content": self._system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        steps: list[ReActStep] = []
        nudged = False  # only nudge once, to avoid looping forever on a stubborn model

        for iteration in range(1, MAX_ITERATIONS + 1):
            data = self._call_model(messages)
            message = data.get("message", {})
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                # Before accepting this as a genuine final answer, check
                # whether the model instead narrated a tool call as plain
                # text (see _extract_text_tool_call) -- otherwise that text
                # would be silently returned as "the answer" and the tool
                # the model clearly intended to run would never execute.
                content = message.get("content", "")
                fallback_call = _extract_text_tool_call(content)
                if fallback_call is None:
                    # Second failure mode: the model merely TALKS about which
                    # tool it could use (e.g. "you can use list_directory to
                    # check...") instead of either answering plainly or
                    # actually calling it. Give it one chance to correct
                    # course before accepting the answer as final.
                    if not nudged and not steps and _mentions_tool_without_calling(content):
                        nudged = True
                        messages.append(message)
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "You described a tool instead of actually calling it. "
                                    "If you need to verify something, call the tool now via "
                                    "the tool-calling mechanism -- do not just describe it."
                                ),
                            }
                        )
                        continue
                    # The model decided it has enough info -- this IS the final answer.
                    return ReActResult(
                        answer=content,
                        steps=steps,
                        iterations=iteration,
                        model=self._model,
                    )
                tool_calls = [{"function": fallback_call}]

            # The model chose to act. Record its own assistant turn, then
            # execute every tool call it asked for and feed results back.
            messages.append(message)
            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                if isinstance(args, str):  # some models emit args as a JSON string
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                tool = _TOOLS_BY_NAME.get(name)
                if tool is None:
                    observation = f"Error: unknown tool '{name}'"
                else:
                    try:
                        observation = str(tool.invoke(args))
                    except Exception as exc:  # defensive: never let a bad tool call crash the loop
                        observation = f"Error running tool '{name}': {exc}"

                steps.append(ReActStep(tool_name=name, tool_args=args, observation=observation))
                messages.append({"role": "tool", "content": observation})

        # Safety valve: the model kept wanting to call tools past our limit.
        return ReActResult(
            answer=(
                "I wasn't able to settle on a final answer within the allowed "
                f"number of steps ({MAX_ITERATIONS})."
            ),
            steps=steps,
            iterations=MAX_ITERATIONS,
            model=self._model,
            hit_iteration_limit=True,
        )
