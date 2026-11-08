"""A tool-calling ReAct agent that works WITH DeepSeek-R1 (reasoning models).

``app.reasoning`` deliberately sends NO tools to DeepSeek-R1, because
Ollama's native ``tools``/``tool_calls`` JSON mechanism (the same one
``app.react_agent.ReActAgent`` uses) turned out, on live testing, to be
unreliable for this model: given a real ``calculator`` tool it ignored it
and did the arithmetic itself (getting it wrong despite correct-looking
reasoning), and given a ``run_shell_command`` tool it never emitted a real
``tool_calls`` field at all -- just prose narrating what the command would
hypothetically return.

That does NOT mean this model can't reliably use tools at all -- it means
the native ``tool_calls`` *JSON-schema fine-tuning* specifically is
unreliable here. So this module uses a DIFFERENT, older protocol instead:
the original ReAct-paper prompting style (predating "function calling"
APIs), where the system prompt instructs the model, in plain text, to emit

    Action: <tool_name>
    Action Input: {"arg_name": "value"}

which THIS module parses itself with a regex and executes for real, feeding
the real result back as the next turn's "Observation:" message -- instead
of depending on whatever tool-calling fine-tuning (or lack of it) the model
happens to have. This only requires decent instruction-following, which
DeepSeek-R1 clearly has (its own `thinking` field shows genuinely correct
step-by-step reasoning).

This was verified empirically against deepseek-r1:14b before writing this
module, in three stages:
  1. Given "What is 847 * 23?" with instructions to use a calculator tool
     via this text format, it reliably emitted a clean, parseable
     ``Action: calculator`` / ``Action Input: {...}`` block.
  2. Left unconstrained, the model kept generating past the Action Input
     line and fabricated its OWN fake Observation/Final Answer. Passing
     Ollama's ``options.stop`` request parameter
     (``["Observation", "\\nObservation"]``) reliably cuts generation off
     immediately after the Action Input line, fixing this.
  3. Fed back a REAL observation (e.g. "Observation: 142" for a file count
     it could not have known), the model's Final Answer correctly used that
     real value -- not a hallucinated one.

KNOWN REMAINING FAILURE MODE -- read carefully before trusting this for
anything destructive: for simple, single-argument tools (calculator,
read_file after one self-correction) this protocol is reliable. For
MULTI-argument, multi-step tasks (e.g. "read this file, then edit it"), live
testing also caught the model skipping the second tool call ENTIRELY and
emitting "Final Answer: <file> has been successfully edited" WITHOUT ever
calling ``edit_file`` -- the file was verifiably untouched on disk. This is
a worse failure mode than simply hitting the iteration limit, because the
textual answer is a confident, false claim of a side effect that never
happened. Mitigation attempted here: the "Action Input" omission is now
caught before invoking the tool (see the ``skeleton`` retry message below),
which measurably helped the model fill in arguments correctly on retry --
but it does NOT guarantee the model will attempt the call at all on a given
turn. Practical takeaway: for multi-step/destructive tool chains with this
model, always verify the actual side effect yourself (e.g. re-read the file)
rather than trusting ``result.answer`` alone -- or use ``app.react_agent``
with ``devstral:24b`` instead, which does not reasoning but has NOT shown
this specific hallucinated-success behavior in testing.

So, unlike ``app.reasoning`` (plain chat, no tools at all, by design), this
module gives you DeepSeek-R1's reasoning AND reliable tool use from the same
model family -- at the cost of needing our own small text-protocol parser
instead of a one-line ``tools=[...]`` kwarg.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import httpx

from app.config import OLLAMA_BASE_URL, OLLAMA_REASONING_MODEL
from app.prompts import DEFAULT_COMPANY_NAME, render_reasoning_react_system_prompt
from app.tools import AVAILABLE_TOOLS

MAX_ITERATIONS = 5  # same hard cap as app.react_agent.ReActAgent
REASONING_REACT_TIMEOUT_SECONDS = 180  # reasoning models are slower than plain chat models

# Stops generation right after the "Action Input: {...}" line, before the
# model can fabricate its own fake Observation/Final Answer -- see point 2
# in the module docstring above.
_STOP_SEQUENCES = ["Observation", "\nObservation"]

_TOOLS_BY_NAME = {t.name: t for t in AVAILABLE_TOOLS}

_ACTION_RE = re.compile(
    r"(?:^|\n)[ \t]*Action:[ \t]*(?P<name>[\w_]+)[ \t]*"
    r"(?:\n+[ \t]*Action Input:[ \t]*(?P<args>\{.*\}))?",
    re.DOTALL | re.MULTILINE,
)
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(?P<answer>.*)", re.DOTALL)


def _build_tool_descriptions() -> str:
    """Render AVAILABLE_TOOLS as a plain-text list the model can read,
    since (unlike the native tool_calls path) this protocol doesn't pass a
    structured JSON schema -- the model only ever sees this text block."""
    lines = []
    for t in AVAILABLE_TOOLS:
        arg_names = ", ".join(t.args.keys()) if t.args else ""
        first_line = (t.description or "").strip().splitlines()[0] if t.description else ""
        lines.append(f"- {t.name}({arg_names}): {first_line}")
    return "\n".join(lines)


@dataclass
class ReasoningReActStep:
    tool_name: str
    tool_args: dict
    observation: str
    reasoning: str = ""  # this step's chain-of-thought, if the model showed one


@dataclass
class ReasoningReActResult:
    answer: str
    reasoning: str = ""  # the FINAL step's chain-of-thought
    steps: list[ReasoningReActStep] = field(default_factory=list)
    iterations: int = 0
    model: str = ""
    hit_iteration_limit: bool = False


class ReasoningReActAgent:
    """ReAct agent for reasoning models, using text-based Action parsing
    instead of Ollama's native tool_calls mechanism (see module docstring)."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str = OLLAMA_BASE_URL,
        company_name: str = DEFAULT_COMPANY_NAME,
    ) -> None:
        self._model = model or OLLAMA_REASONING_MODEL
        self._base_url = base_url.rstrip("/")
        self._system_prompt = render_reasoning_react_system_prompt(
            tool_descriptions=_build_tool_descriptions(),
            company_name=company_name,
        )

    def _call_model(self, messages: list[dict]) -> dict:
        response = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "think": True,
                "options": {"stop": _STOP_SEQUENCES},
            },
            timeout=REASONING_REACT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    def handle(self, prompt: str, history: list[dict] | None = None) -> ReasoningReActResult:
        """Run the text-based ReAct loop for ``prompt``. Same ``history``
        contract as ``app.react_agent.ReActAgent.handle``: an optional list
        of prior {"role", "content"} turns, user-visible text only."""
        messages = [{"role": "system", "content": self._system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        steps: list[ReasoningReActStep] = []

        for iteration in range(1, MAX_ITERATIONS + 1):
            data = self._call_model(messages)
            message = data.get("message", {})
            content = message.get("content", "") or ""
            thinking = message.get("thinking", "") or ""

            action_match = _ACTION_RE.search(content)
            if action_match:
                name = action_match.group("name")
                raw_args = action_match.group("args")
                if raw_args is None:
                    # No "Action Input:" line at all -- expected for a
                    # zero-argument tool like system_resources(), which has
                    # nothing to put there.
                    args = {}
                else:
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {}

                tool = _TOOLS_BY_NAME.get(name)
                if tool is None:
                    observation = f"Error: unknown tool '{name}'"
                elif not args and tool.args:
                    # Regression guard: the model sometimes omits the Action
                    # Input line entirely (or emits "{}") even for a tool
                    # that DOES require arguments, especially for
                    # multi-argument tools like write_file/edit_file. Rather
                    # than invoking the tool (guaranteed to fail with a raw,
                    # unhelpful Pydantic validation dump) and burning an
                    # iteration figuring that out, give a direct, actionable
                    # error naming the exact missing keys -- this measurably
                    # helped the model self-correct within the iteration
                    # budget during live testing.
                    # A literal, copy-shapeable JSON skeleton (not just the
                    # key names as prose) measurably reduced repeated
                    # omissions during live testing on multi-argument tools
                    # like edit_file/write_file, vs. naming the keys alone.
                    skeleton = json.dumps({k: f"<{k}>" for k in tool.args})
                    observation = (
                        f"Error: no Action Input was given for '{name}', but it "
                        f"requires arguments. Re-emit EXACTLY this shape (with "
                        "real values instead of the <placeholders>):\n"
                        f"Action: {name}\n"
                        f"Action Input: {skeleton}"
                    )
                else:
                    try:
                        observation = str(tool.invoke(args))
                    except Exception as exc:  # never let a bad tool call crash the loop
                        observation = f"Error running tool '{name}': {exc}"

                steps.append(
                    ReasoningReActStep(
                        tool_name=name,
                        tool_args=args,
                        observation=observation,
                        reasoning=thinking,
                    )
                )
                # Only replay the model's own text up to (and including) the
                # Action Input line -- never anything past it, in case the
                # stop sequence didn't fire for some reason -- so the model
                # never sees itself having "already" fabricated an answer.
                messages.append(
                    {"role": "assistant", "content": content[: action_match.end()]}
                )
                messages.append({"role": "user", "content": f"Observation: {observation}"})
                continue

            final_match = _FINAL_ANSWER_RE.search(content)
            answer = final_match.group("answer").strip() if final_match else content.strip()

            return ReasoningReActResult(
                answer=answer,
                reasoning=thinking,
                steps=steps,
                iterations=iteration,
                model=self._model,
            )

        return ReasoningReActResult(
            answer=(
                "I wasn't able to settle on a final answer within the allowed "
                f"number of steps ({MAX_ITERATIONS})."
            ),
            steps=steps,
            iterations=MAX_ITERATIONS,
            model=self._model,
            hit_iteration_limit=True,
        )

