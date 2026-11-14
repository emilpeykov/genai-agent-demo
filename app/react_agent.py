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
import os
from dataclasses import dataclass, field

import httpx

from app.config import MODEL_CATALOG, OLLAMA_BASE_URL
from app.prompts import DEFAULT_COMPANY_NAME, render_react_system_prompt
from app.tool_schemas import validate_tool_args
from app.tools import AVAILABLE_TOOLS

# Hard cap on ReAct rounds (1 round = either one batch of tool call(s), or
# the final answer) so a confused model can't loop forever. NOTE: this is
# NOT the same thing as how many TOOL CALLS a simple question needs -- e.g.
# a single-tool question like "how many .py files are here?" normally
# reports iterations=2 (round 1: decide + call the tool, round 2: give the
# final answer using the real observation) long before ever approaching
# this cap; ``ReActResult.hit_iteration_limit`` tells you definitively
# whether THIS cap was actually hit, rather than guessing from the raw
# iteration count. Configurable via REACT_MAX_ITERATIONS since genuinely
# multi-hop tasks (e.g. "list the dir, THEN search each subfolder, THEN
# summarize") can legitimately need more than a handful of rounds.
MAX_ITERATIONS = int(os.getenv("REACT_MAX_ITERATIONS", "8"))
# Bumped from 120s: requesting a bigger num_ctx (see OLLAMA_NUM_CTX below)
# forces Ollama to reload the model with a larger KV cache the first time,
# which measurably takes longer than a normal request.
REACT_TIMEOUT_SECONDS = 180

# BUG FIX (observed: "the model doesn't remember the file path I told it
# earlier"): this was NOT a history-threading bug -- app.gradio_app really
# was sending prior turns back correctly. The actual cause: Ollama loads
# qwen3:14b with a default context window of only 4096 tokens (confirmed via
# `curl localhost:11434/api/ps` -> "context_length": 4096) UNLESS a request
# explicitly asks for more via `options.num_ctx`. Measured empirically: this
# agent's own system prompt + all 17 tool JSON schemas ALONE already total
# roughly ~4300 estimated tokens -- i.e. MORE than the entire default
# context window, BEFORE a single user message, prior turn, or tool
# observation is added. Ollama silently truncates the OLDEST tokens once the
# prompt exceeds num_ctx, so anything said earlier in the conversation (e.g.
# "always save memory usage to /tmp/x.txt") was being dropped before the
# model ever saw it again -- not forgotten by the model, never actually
# delivered to it. Explicitly requesting a much larger context window below
# fixes this. Configurable since it trades off VRAM/latency for context size.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))

_TOOLS_BY_NAME = {t.name: t for t in AVAILABLE_TOOLS}

# ---------------------------------------------------------------------------
# Deterministic "search escalation" nudge -- a CODE-LEVEL complement to the
# prompt instruction in app.prompts (REACT_SYSTEM_TEMPLATE's "SEARCH
# ESCALATION" section). Prompt text alone is a request the model MAY ignore
# (exactly as observed: a thin 'web_search' result was accepted as final
# without ever trying 'google_search'/'fetch_webpage'). This mirrors the
# project's existing pattern of not trusting the model to reliably do
# something itself (see _extract_touched_files/_known_files_message above
# for the same "don't just ask nicely in the prompt, verify/enforce in code"
# approach to file-path memory): after EVERY search-tool call, the real
# tool_calls loop inspects the observation and, if it looks thin/empty/error,
# deterministically injects a system-level reminder naming the SPECIFIC
# search tools not yet tried THIS turn -- forcing the next model call to see
# a concrete, unavoidable nudge instead of relying purely on it having
# "remembered" the system prompt's instruction.
# ---------------------------------------------------------------------------
SEARCH_TOOL_NAMES: tuple[str, ...] = (
    "web_search",
    "google_search",
    "wikipedia_search",
    "arxiv_search",
    "stackexchange_search",
)

# Name of the human-in-the-loop tool (see its docstring in app.tools for the
# full rationale) -- calling it must deterministically STOP the ReAct loop
# right there and surface the question to the caller, rather than letting
# the model keep "acting" past a question it asked the user and has no
# answer to yet. Checked by name (not by sniffing the tool's return text)
# everywhere a tool call is executed in this module and its sibling agents.
ASK_USER_TOOL_NAME = "ask_user"


def is_thin_search_result(observation: str) -> bool:
    """True if a search tool's observation looks empty, erroring, or
    otherwise unhelpful -- i.e. NOT a real answer the agent should settle
    for without trying another source first.

    Deliberately pattern-based (generic "Error"/"No ... found" prefixes)
    rather than hardcoding each tool's exact wording, so it keeps working
    if a tool's message text changes or a new search tool is added later.

    Deliberately does NOT try to guess relevance/accuracy by keyword-
    matching the query against the result text -- that's a crude heuristic
    prone to false positives/negatives. Instead, non-empty/non-error
    results are handed to the LLM itself via ``search_relevance_hint``
    below, which asks the model to judge accuracy and decide whether to
    search again -- a real judgment call, not a regex guess.
    """
    stripped = observation.strip()
    if not stripped:
        return True
    if stripped.startswith("Error"):
        return True
    lowered = stripped.lower()
    return lowered.startswith("no ") and "found" in lowered


def search_escalation_hint(tool_name: str, tried_search_tools: set[str]) -> dict | None:
    """Build the system-level reminder message to inject right after a thin
    (empty/error/"no results") search result, naming the search tools NOT
    yet tried this turn. Returns None once every search tool has already
    been tried (nothing new to suggest -- the model must decide on its own
    at that point)."""
    remaining = [t for t in SEARCH_TOOL_NAMES if t not in tried_search_tools]
    if not remaining:
        return None
    return {
        "role": "system",
        "content": (
            f"SEARCH ESCALATION REMINDER: '{tool_name}' just returned a "
            "thin/empty/error result -- that is NOT proof the information "
            f"doesn't exist, it just means that one source didn't have it. "
            f"You have NOT yet tried: {', '.join(remaining)}. Call one of "
            "those next (and 'fetch_webpage' on any promising result URL "
            "to read the full page) before concluding this can't be "
            "answered."
        ),
    }


def search_relevance_hint(
    tool_name: str, query: str, tried_search_tools: set[str]
) -> dict | None:
    """Build the system-level reminder injected after a search result that
    is NOT empty/erroring (so ``is_thin_search_result`` wouldn't flag it),
    asking the LLM itself to judge whether that result is actually accurate
    and on-topic before relying on it.

    BUG FIX (observed live): escalating from a failed 'google_search' to
    'wikipedia_search' worked, but 'wikipedia_search' then matched a
    completely UNRELATED Wikipedia article (e.g. query "OpenKBS agentic
    services" returned a biography of a South Korean actress) -- a
    non-empty, non-error result that ``is_thin_search_result`` has no way
    to tell is simply about the wrong subject. Rather than hardcoding a
    keyword-overlap guess (brittle, easy to get wrong both ways), this
    hands the decision back to the model: it just saw the real
    observation, so it is far better positioned than any regex to judge
    whether it genuinely answers the ORIGINAL query -- this message just
    makes that judgment call an explicit, required step instead of an
    implicit hope, and names the untried tools in case the model decides
    to search again.
    """
    remaining = [t for t in SEARCH_TOOL_NAMES if t not in tried_search_tools]
    next_tools = (
        f"You have not yet tried: {', '.join(remaining)}."
        if remaining
        else "You have already tried every available search tool."
    )
    return {
        "role": "system",
        "content": (
            f"SEARCH RESULT CHECK: '{tool_name}' just returned a result for "
            f"the query '{query}'. Before using it, judge for yourself "
            "whether it ACTUALLY and ACCURATELY answers what was asked -- "
            "check it is really about the right subject, not just "
            "superficially similar wording (e.g. a search can return a "
            "real, detailed result that is nonetheless about the WRONG "
            "person/company/topic). If it looks accurate and on-topic, go "
            "ahead and use it. If it looks irrelevant, incomplete, or "
            f"you're not confident it's correct, do NOT use it -- {next_tools} "
            "Call one of those next (or 'fetch_webpage' on a promising "
            "result URL to read the full page) instead of settling for an "
            "inaccurate result."
        ),
    }


# Which argument(s) of each file-touching tool hold a REAL file path, used by
# ``_extract_touched_files`` below to build a deterministic, code-guaranteed
# memory of every exact path seen so far in a conversation -- see that
# function's docstring for why this exists (prompt instructions alone proved
# unreliable: an LLM can still choose not to restate a path in its own prose
# final answer, and that answer text is the ONLY thing replayed into later
# turns' history, so the path would otherwise be silently, permanently lost).
_FILE_PATH_ARGS_BY_TOOL: dict[str, tuple[str, ...]] = {
    "write_file": ("file_path",),
    "read_file": ("file_path",),
    "file_delete": ("file_path",),
    "edit_file": ("path",),
    "copy_file": ("source_path", "destination_path"),
    "move_file": ("source_path", "destination_path"),
}


def _extract_touched_files(steps: list[ReActStep]) -> list[str]:
    """Deterministically collect every real file path seen in ``steps``'
    tool calls (write_file/read_file/edit_file/copy_file/move_file/
    file_delete), in first-seen order, de-duplicated.

    BUG FIX (observed w/ qwen3:14b): asked to create a file in one turn, then
    "read that file back" in a LATER turn, the model had no memory of the
    exact path -- not because resending conversation history doesn't work
    (it does -- see OLLAMA_NUM_CTX above), but because ``handle()``/
    ``handle_stream()`` only ever replay each past turn's plain ANSWER TEXT
    into later requests, never the raw tool_calls/arguments behind it. If the
    model's own answer was vague ("I've saved it for you") instead of
    literally restating the path, that path was simply never present in ANY
    message sent back to the model on the next turn -- a real gap in what
    our own code resends, not a prompting problem the model could reliably
    work around. This function lets a caller (e.g. app.gradio_app) track
    real paths touched across turns ITSELF, independent of what the model
    chooses to say, and feed them back deterministically via ``handle()``'s
    ``known_files`` parameter below.
    """
    seen: list[str] = []
    for step in steps:
        arg_names = _FILE_PATH_ARGS_BY_TOOL.get(step.tool_name)
        if not arg_names:
            continue
        for arg_name in arg_names:
            value = step.tool_args.get(arg_name)
            if isinstance(value, str) and value and value not in seen:
                seen.append(value)
    return seen


def _known_files_message(known_files: list[str]) -> dict:
    """Build the deterministic system-level reminder message for
    ``known_files`` -- real paths touched earlier in the conversation,
    tracked in CODE (see _extract_touched_files), not dependent on whether
    the model's own past answers happened to mention them."""
    listing = "\n".join(f"- {path}" for path in known_files)
    return {
        "role": "system",
        "content": (
            "FILES ALREADY TOUCHED EARLIER IN THIS CONVERSATION (exact real "
            "paths, tracked automatically regardless of how they were "
            "described in chat -- use these exact paths for any follow-up "
            "request like 'read that file', 'edit it', 'that file you "
            "made', etc., instead of guessing or searching for the most "
            "recently modified file):\n" + listing
        ),
    }



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


def execute_tool_call(call: dict) -> ReActStep:
    """Run one native Ollama ``tool_calls`` entry and return the resulting
    ``ReActStep``. Free function (no ``self`` needed) so it can be reused
    verbatim by ``ReActAgent._execute_tool_call`` AND by
    ``app.react_langgraph_agent.ReActLangGraphAgent``'s LangGraph node,
    instead of the same validation/execution logic drifting between two
    copies."""
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
        # Client-side validation pass: catch malformed arguments (wrong
        # type, missing required field, etc.) BEFORE they reach the real
        # tool function, and feed a precise pydantic validation error
        # back as the observation instead of a generic exception message
        # -- this gives the model (on local/smaller tool-calling models
        # especially) a concrete, actionable signal to self-correct the
        # SAME tool call on its very next turn, rather than just
        # learning "that failed" with no clue why. See app.tool_schemas
        # for the full rationale and the per-tool schemas themselves.
        validated_args, validation_error = validate_tool_args(name, args)
        if validation_error:
            observation = validation_error
        else:
            try:
                observation = str(tool.invoke(validated_args))
            except Exception as exc:  # defensive: never let a bad tool call crash the loop
                observation = f"Error running tool '{name}': {exc}"
    return ReActStep(tool_name=name, tool_args=args, observation=observation)


@dataclass
class ReActResult:
    answer: str
    steps: list[ReActStep] = field(default_factory=list)
    iterations: int = 0
    model: str = ""
    hit_iteration_limit: bool = False
    # Deterministically extracted from this turn's OWN steps (see
    # _extract_touched_files) -- NOT dependent on whether the model's
    # ``answer`` text happened to mention them. A caller that wants file
    # memory across turns should accumulate these (e.g. a running
    # dict/list keyed by path) and pass the combined list back in as
    # ``handle()``'s ``known_files`` argument on the NEXT call.
    touched_files: list[str] = field(default_factory=list)
    # True if the loop stopped because the model called 'ask_user' -- in
    # that case ``answer`` IS the question to show the real user, not a
    # final answer. The caller should display it, collect a reply, and call
    # handle()/handle_stream() again with that reply as the next ``prompt``
    # (same history/thread_id as before) to resume the loop where it left
    # off -- see app.tools.ask_user's docstring for the full rationale.
    awaiting_user_input: bool = False



class ReActAgent:
    """LLM-driven tool-calling agent using a local Ollama model."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str = OLLAMA_BASE_URL,
        company_name: str = DEFAULT_COMPANY_NAME,
    ) -> None:
        self._model = model or MODEL_CATALOG["balanced"].ollama_model
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
                "options": {"num_ctx": OLLAMA_NUM_CTX},
            },
            timeout=REACT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    def _call_model_stream(self, messages: list[dict]):
        """Same request as ``_call_model``, but with ``stream: True`` --
        yields each raw NDJSON chunk Ollama sends as it's generated (one
        dict per line), instead of waiting for the whole response. Verified
        empirically (see app.gradio_app) that for qwen3:14b: "thinking" and
        "content" each arrive incrementally, token-by-token, across many
        chunks, while "tool_calls" (when present) arrives whole in a single
        chunk -- Ollama does not stream a tool call's JSON piecemeal."""
        with httpx.stream(
            "POST",
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "tools": self._tool_schemas,
                "stream": True,
                "options": {"num_ctx": OLLAMA_NUM_CTX},
            },
            timeout=REACT_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                yield json.loads(line)

    def _execute_tool_call(self, call: dict) -> ReActStep:
        """Shared by ``handle`` and ``handle_stream`` so the actual
        tool-execution logic can't drift between the two code paths.

        Delegates to the module-level ``execute_tool_call`` function below --
        pulled out to a free function (it never used ``self``) so
        ``app.react_langgraph_agent``'s LangGraph node can reuse the EXACT
        same tool-execution/validation logic instead of duplicating it."""
        return execute_tool_call(call)

    def handle(
        self,
        prompt: str,
        history: list[dict] | None = None,
        known_files: list[str] | None = None,
    ) -> ReActResult:
        """Run the ReAct loop for ``prompt``.

        ``history`` is an optional list of prior ``{"role": ..., "content": ...}``
        turns (roles "user"/"assistant") from earlier in the conversation --
        without it, every call is completely independent and the model has no
        memory of anything said before (e.g. it won't know what "it" refers
        to in a follow-up question). Tool-call/observation turns from earlier
        exchanges are intentionally NOT replayed here (they'd bloat the
        context very fast); only the user-visible text of each turn is kept.

        ``known_files`` is an optional list of real file paths touched in
        EARLIER turns (accumulate ``ReActResult.touched_files`` across calls
        and pass the running list back in here) -- since ``history`` above
        only carries each past turn's plain answer TEXT, a vague past answer
        ("I've saved it for you") would otherwise make an exact path
        permanently unrecoverable on a later turn. Passing it here injects a
        deterministic system-level reminder listing the real paths,
        independent of whether the model's own past answer happened to state
        them.
        """
        messages = [{"role": "system", "content": self._system_prompt}]
        if history:
            messages.extend(history)
        if known_files:
            messages.append(_known_files_message(known_files))
        messages.append({"role": "user", "content": prompt})
        steps: list[ReActStep] = []
        tried_search_tools: set[str] = set()

        for iteration in range(1, MAX_ITERATIONS + 1):
            data = self._call_model(messages)
            message = data.get("message", {})
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                # The model decided it has enough info -- this IS the final answer.
                return ReActResult(
                    answer=message.get("content", ""),
                    steps=steps,
                    iterations=iteration,
                    model=self._model,
                    touched_files=_extract_touched_files(steps),
                )


            # The model chose to act. Record its own assistant turn, then
            # execute every tool call it asked for and feed results back.
            #
            # BUG FIX (observed w/ qwen3:14b, a reasoning-capable tool-caller):
            # the raw message dict Ollama returns includes a "thinking" field
            # (qwen3 emits chain-of-thought even without --think, unlike
            # DeepSeek-R1 which app.reasoning_react_agent asks for explicitly).
            # Replaying that "thinking" text back verbatim as part of this
            # assistant turn's history -- which naive ``messages.append(message)``
            # does -- measurably confused qwen3 on the VERY NEXT turn: live
            # testing showed it would then treat the following tool
            # observation (e.g. "8035") as a brand-new, context-free user
            # message and ask for clarification on "what is 8035?", instead of
            # using it to answer the original question. Qwen's own guidance
            # (unlike DeepSeek-R1's) is that prior turns' thinking should NOT
            # be kept in multi-turn history -- stripping it here fixed the
            # behavior in live testing. The final answer's own "thinking" is
            # still available via the LAST raw ``message`` for callers that
            # want to show chain-of-thought for the CURRENT turn; only what we
            # replay back into the model's own context is affected.
            messages.append({k: v for k, v in message.items() if k != "thinking"})
            for call in tool_calls:
                step = self._execute_tool_call(call)
                steps.append(step)
                # BUG FIX (observed w/ qwen3:14b): Ollama's tool_calls responses
                # include a per-call "id" (e.g. "call_b2c0uisi"), used by the
                # model's own chat template to associate a later "tool" message
                # with THE SPECIFIC call that produced it -- the same contract
                # OpenAI's function-calling API uses via "tool_call_id". We were
                # dropping that id and sending back a bare {"role": "tool",
                # "content": ...} message. Without the link, qwen3 didn't
                # recognize the next turn's "tool" message as a reply to ITS
                # OWN request -- it instead treated the bare observation text
                # (e.g. "8035") as a brand-new, unrelated user input and asked
                # for clarification on it, instead of using it to answer the
                # original question. Echoing "tool_call_id" back fixes this.
                tool_message = {"role": "tool", "content": step.observation}
                if call.get("id"):
                    tool_message["tool_call_id"] = call["id"]
                messages.append(tool_message)

                # Human-in-the-loop short-circuit: stop the WHOLE loop right
                # here, surfacing the question as this turn's answer, rather
                # than continuing to execute any remaining tool calls in this
                # same batch or looping back to the model for another round
                # it has no real answer to give yet. See ASK_USER_TOOL_NAME's
                # comment above / app.tools.ask_user's docstring.
                if step.tool_name == ASK_USER_TOOL_NAME:
                    return ReActResult(
                        answer=step.tool_args.get("question", step.observation),
                        steps=steps,
                        iterations=iteration,
                        model=self._model,
                        touched_files=_extract_touched_files(steps),
                        awaiting_user_input=True,
                    )

                # Deterministic search-escalation nudge -- see
                # search_escalation_hint's/search_relevance_hint's docstrings
                # above. Obviously thin/empty/error results get a direct
                # escalation reminder; anything else is handed back to the
                # LLM itself with an explicit "is this actually accurate?"
                # check instead of us guessing relevance by keyword overlap.
                if step.tool_name in SEARCH_TOOL_NAMES:
                    tried_search_tools.add(step.tool_name)
                    if is_thin_search_result(step.observation):
                        hint = search_escalation_hint(step.tool_name, tried_search_tools)
                    else:
                        hint = search_relevance_hint(
                            step.tool_name, step.tool_args.get("query", ""), tried_search_tools
                        )
                    if hint:
                        messages.append(hint)
        return ReActResult(
            answer=(
                "I wasn't able to settle on a final answer within the allowed "
                f"number of steps ({MAX_ITERATIONS})."
            ),
            steps=steps,
            iterations=MAX_ITERATIONS,
            model=self._model,
            hit_iteration_limit=True,
            touched_files=_extract_touched_files(steps),
        )

    def handle_stream(
        self,
        prompt: str,
        history: list[dict] | None = None,
        known_files: list[str] | None = None,
    ):
        """Streaming counterpart to ``handle()``, for UIs that want to show
        the model's output as it's generated (e.g. app.gradio_app) instead
        of waiting for the whole ReAct loop to finish.

        Yields dicts (NOT a ReActResult) describing what's happening live:
          - {"type": "thinking_delta", "text": ...}  -- a chunk of the
            CURRENT iteration's chain-of-thought, as it streams in.
          - {"type": "content_delta", "text": ...}  -- a chunk of the
            CURRENT iteration's answer text (only meaningful on the final,
            no-tool-calls iteration -- see module docstring note below).
          - {"type": "tool_call", "tool_name": ..., "tool_args": ...}  --
            the model decided to call a tool (arrives as one whole chunk,
            not streamed piecemeal -- see _call_model_stream).
          - {"type": "observation", "tool_name": ..., "observation": ...}
            -- the REAL result of running that tool.
          - {"type": "final", "result": ReActResult}  -- the loop is done;
            same shape ``handle()`` returns, for metadata (model, iteration
            count, hit_iteration_limit, all steps, touched_files).

        ``known_files`` -- see ``handle()``'s docstring for the full
        rationale: a deterministic, code-tracked list of real file paths
        touched in earlier turns, injected as a system-level reminder so a
        follow-up like "read that file back" works even if the model's own
        past answer was too vague to carry the exact path forward on its own.
        """
        messages = [{"role": "system", "content": self._system_prompt}]
        if history:
            messages.extend(history)
        if known_files:
            messages.append(_known_files_message(known_files))
        messages.append({"role": "user", "content": prompt})
        steps: list[ReActStep] = []
        tried_search_tools: set[str] = set()


        for iteration in range(1, MAX_ITERATIONS + 1):
            content_acc = ""
            thinking_acc = ""
            tool_calls: list[dict] = []

            for chunk in self._call_model_stream(messages):
                chunk_message = chunk.get("message", {})
                thinking_piece = chunk_message.get("thinking") or ""
                content_piece = chunk_message.get("content") or ""
                if thinking_piece:
                    thinking_acc += thinking_piece
                    yield {"type": "thinking_delta", "text": thinking_piece}
                if content_piece:
                    content_acc += content_piece
                    yield {"type": "content_delta", "text": content_piece}
                if chunk_message.get("tool_calls"):
                    tool_calls = chunk_message["tool_calls"]

            if not tool_calls:
                # Same early-return condition as handle(): no tool calls
                # means the model considers content_acc its final answer.
                result = ReActResult(
                    answer=content_acc,
                    steps=steps,
                    iterations=iteration,
                    model=self._model,
                    touched_files=_extract_touched_files(steps),
                )
                yield {"type": "final", "result": result}
                return

            # Same history-shaping fix as handle(): never replay "thinking"
            # back into the model's own context (see the long comment on
            # that line in handle() above for why).
            messages.append(
                {"role": "assistant", "content": content_acc, "tool_calls": tool_calls}
            )
            for call in tool_calls:
                fn = call.get("function", {})
                yield {
                    "type": "tool_call",
                    "tool_name": fn.get("name", ""),
                    "tool_args": fn.get("arguments", {}),
                }
                step = self._execute_tool_call(call)
                steps.append(step)
                yield {
                    "type": "observation",
                    "tool_name": step.tool_name,
                    "observation": step.observation,
                }
                tool_message = {"role": "tool", "content": step.observation}
                if call.get("id"):
                    tool_message["tool_call_id"] = call["id"]
                messages.append(tool_message)

                # Human-in-the-loop short-circuit -- same rationale as in
                # handle() above.
                if step.tool_name == ASK_USER_TOOL_NAME:
                    result = ReActResult(
                        answer=step.tool_args.get("question", step.observation),
                        steps=steps,
                        iterations=iteration,
                        model=self._model,
                        touched_files=_extract_touched_files(steps),
                        awaiting_user_input=True,
                    )
                    yield {"type": "final", "result": result}
                    return

                if step.tool_name in SEARCH_TOOL_NAMES:
                    tried_search_tools.add(step.tool_name)
                    if is_thin_search_result(step.observation):
                        hint = search_escalation_hint(step.tool_name, tried_search_tools)
                    else:
                        hint = search_relevance_hint(
                            step.tool_name, step.tool_args.get("query", ""), tried_search_tools
                        )
                    if hint:
                        messages.append(hint)

        result = ReActResult(
            answer=(
                "I wasn't able to settle on a final answer within the allowed "
                f"number of steps ({MAX_ITERATIONS})."
            ),
            steps=steps,
            iterations=MAX_ITERATIONS,
            model=self._model,
            hit_iteration_limit=True,
            touched_files=_extract_touched_files(steps),
        )
        yield {"type": "final", "result": result}


