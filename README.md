# GenAI Agent Demo

A small, production-shaped FastAPI service demonstrating core **AI Engineer**
skills: LLM agent orchestration, prompt/behavior design, model routing with
cost/latency tradeoffs, and failure-mode mitigation — runnable entirely
offline via a built-in mock LLM provider (no API key required).

## Why this project exists

This was scaffolded to directly demonstrate the skills listed in a real AI
Engineer job posting:

| Requirement | Where it's demonstrated |
|---|---|
| LLMs, prompt engineering, agent frameworks (LangChain/LlamaIndex/AutoGen) | `app/tools.py` (LangChain `@tool`s), `app/agent.py` (tool-vs-LLM decision loop) |
| Translating ambiguous business goals into behavioral LLM design | `app/agent.py` routing rules (when to use a deterministic tool vs. an LLM, and which tier) |
| Model routing, cost/latency tradeoffs | `app/config.py` (model tier catalog with cost/latency), `app/router.py` (complexity classifier → tier) |
| Failure-mode mitigation in production GenAI apps | `app/resilience.py` (retry with backoff + tier fallback chain) |
| Python, API development | `app/main.py` (FastAPI service, typed request/response models) |
| Git, unit testing, CI/CD | `tests/` (pytest suite), `.github/workflows/ci.yml` (GitHub Actions) |
| Data disciplines (MLOps-adjacent) | Structured telemetry per request: model, tier, latency, cost, retries |
| Reasoning-model chain-of-thought | `app/reasoning.py` (DeepSeek-R1 via Ollama, `/chat/reasoning`) |

## Architecture

```
Request ("prompt")
      │
      ▼
┌─────────────┐   deterministic match    ┌────────────────────┐
│   Agent     │ ───────────────────────▶ │ LangChain tools     │  (calculator,
│ (app/agent) │                          │ (app/tools.py)      │   KB lookup)
└─────┬───────┘                          └────────────────────┘
      │ no tool match
      ▼
┌─────────────────┐   tier (fast/       ┌───────────────────────┐
│ Complexity       │   balanced/        │ Resilience layer       │
│ router           │ ─ powerful) ──────▶│ (retry + fallback)     │
│ (app/router.py)  │                    │ (app/resilience.py)    │
└─────────────────┘                    └──────────┬─────────────┘
                                                   ��
                                        ┌───────────────────────┐
                                        │ LLM Provider           │
                                        │ (mock or OpenAI/Chat)  │
                                        │ (app/providers.py)     │
                                        └───────────────────────┘
```

**Routing logic**: short/simple prompts go to the cheapest "fast" tier;
prompts needing synthesis (summarize/explain) go to "balanced"; prompts with
multi-step reasoning signals or long context go to "powerful". Deterministic
sub-tasks (arithmetic, known internal facts) bypass the LLM entirely via
tools — the cheapest and most reliable "model" is the one you don't call.

**Resilience**: each tier is retried with exponential backoff on transient
failures; if a tier is exhausted, the request automatically falls back to the
next cheaper tier rather than failing outright (with the response flagged
`degraded: true` so this is observable).

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # leave OPENAI_API_KEY blank to use the mock provider
uvicorn app.main:app --reload
```

Try it:

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"prompt": "Design a step-by-step algorithm to reduce our LLM costs."}'
```

Response includes routing/cost telemetry:

```json
{
  "answer": "[in-depth analysis] Response to: Design a step-by-step algorithm...",
  "tool_used": null,
  "model": "o1",
  "tier": "powerful",
  "latency_ms": 2104,
  "estimated_cost_usd": 0.000495,
  "retries": 0,
  "degraded": false,
  "routing_reason": "Prompt contains multi-step/reasoning signals or is long."
}
```

## Running tests

```bash
pytest -v
```

## Using a real LLM instead of the mock

1. `pip install langchain-openai`
2. Set `OPENAI_API_KEY` in `.env`.
3. The app automatically switches to `OpenAIChatProvider` — no code changes
   needed, since both providers implement the same `LLMProvider` interface.

## Reasoning model (DeepSeek-R1 via Ollama)

Besides the fast/balanced/powerful cost-tier routing above, the app also
exposes a dedicated path for **reasoning models** — local models (e.g.
DeepSeek-R1, QwQ) trained to emit a visible chain-of-thought *before* their
final answer, instead of jumping straight to a response. This is a
genuinely different, inspectable capability (useful for debugging *why* a
model reached a conclusion, or building "show your work" UIs), not just a
bigger/slower model in the same family — so it's deliberately kept separate
from `MODEL_CATALOG` in `app/config.py`.

```bash
ollama pull deepseek-r1:14b   # ~9 GB; use :32b instead if you have the RAM/VRAM
```

```bash
LLM_BACKEND=ollama uvicorn app.main:app --reload
curl -X POST http://127.0.0.1:8000/chat/reasoning -H "Content-Type: application/json" \
  -d '{"prompt": "Is 1009 a prime number? Show your reasoning."}'
```

```json
{
  "reasoning": "1009 is odd, not divisible by 3, 7, 11, 13... up to sqrt(1009)≈31.8, so check primes up to 31 ...",
  "answer": "Yes, 1009 is a prime number.",
  "model": "deepseek-r1:14b",
  "latency_ms": 8421
}
```

A streaming NDJSON variant is available at `/chat/reasoning/stream` (each
line is `{"kind": "reasoning"|"answer", "text": "..."}`), and both are also
available as a "Reasoning model" tab in the Gradio UI
(`python -m app.gradio_app`), rendering the chain-of-thought in a collapsible
section above the final answer.

Set `OLLAMA_REASONING_MODEL` (see `.env.example`) to use a different model,
e.g. `deepseek-r1:32b` for higher quality at the cost of more RAM/VRAM and
latency. Reasoning models legitimately take much longer than a normal
completion to respond — that's expected "thinking" time, not a hang.

**No tool use in `/chat/reasoning`, on purpose.** It's pure chat completion
(no tools wired in). This was tested, not assumed: `ollama show deepseek-r1:14b`
lists `tools` as a capability, but in practice the model does NOT reliably
act on Ollama's *native* `tools`/`tool_calls` JSON mechanism — given a real
`calculator` tool via that mechanism it ignored it and did arithmetic in its
own "thinking" (getting it wrong), and given a `run_shell_command` tool for
something it genuinely couldn't know, it never emitted a real tool call at
all — it just narrated, in prose, what running the command would
hypothetically return, complete with a fabricated example value.

**But DeepSeek-R1 CAN reliably use tools, via a different protocol.** That
native-mechanism failure is specific to how the model was fine-tuned, not a
sign it can't follow instructions to use a tool. `/agent/react/reasoning`
(via `app.reasoning_react_agent.ReasoningReActAgent`) uses the ORIGINAL
ReAct-paper prompting style instead (predating "function calling" APIs): the
system prompt instructs the model, in plain text, to emit

```
Action: calculator
Action Input: {"expression": "847 * 23"}
```

which we parse ourselves with a regex, execute for real, and feed the real
result back as an `Observation:` message. This was verified empirically to
work reliably — live-tested against the running model:

```bash
curl -X POST http://127.0.0.1:8000/agent/react/reasoning -H "Content-Type: application/json" \
  -d '{"prompt": "What is 847 * 23? Use the calculator tool."}'
```

```json
{
  "answer": "The product of 847 and 23 is 19,481.",
  "reasoning": "... chain-of-thought ...",
  "steps": [
    {"tool_name": "calculator", "tool_args": {"expression": "847 * 23"}, "observation": "19481", "reasoning": "..."}
  ],
  "iterations": 2,
  "model": "deepseek-r1:14b",
  "hit_iteration_limit": false
}
```

Note the real tool call and the *correct* answer (19,481) — a direct
improvement over the native-mechanism test above, which silently miscalculated
instead. Also available as a "Reasoning + tools" tab in the Gradio UI.

**Same thing, orchestrated with LangGraph instead of a `for` loop.**
`/agent/react/reasoning/langgraph` (via
`app.reasoning_langgraph_agent.ReasoningLangGraphAgent`) is the exact same
model, system prompt, and text-based `Action:`/`Action Input:` protocol —
but as an explicit LangGraph `StateGraph` with a dedicated
**self-correction edge**: a genuinely malformed tool call (invalid JSON in
Action Input) is caught by a conditional edge that routes back to the
model with the *real* `json.JSONDecodeError` message as feedback, instead
of silently falling back to empty args. Reasoning models like DeepSeek-R1
often need exactly this kind of retry — a straight-line pipeline has
nowhere for a parse failure to go but raise or degrade; LangGraph's graph
supports the cycle natively. Also available as a "Reasoning + tools,
LangGraph" tab in the Gradio UI.

## Possible next steps (not implemented, good talking points)

- Swap the heuristic complexity classifier for a small fine-tuned/few-shot
  classifier, or an actual cheap-LLM-as-judge call.
- Add LlamaIndex-based RAG over real documents in place of the mock KB tool.
- Add an AutoGen-style critic agent that reviews the primary agent's answer
  before returning it for high-stakes tiers.
- Add tracing (e.g. LangSmith/OpenTelemetry) and a cost dashboard.
