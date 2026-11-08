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

**No tool use here, on purpose.** `/chat/reasoning` is pure chat completion
(no tools wired in), unlike `/chat/lcel/tools` or `/agent/react`. This was
tested, not assumed: `ollama show deepseek-r1:14b` lists `tools` as a
capability, but in practice the model does NOT reliably act on it — given a
real `calculator` tool it ignored it and did arithmetic in its own
"thinking" (getting it wrong), and given a `run_shell_command` tool for
something it genuinely couldn't know, it never emitted a real tool call at
all — it just narrated, in prose, what running the command would
hypothetically return, complete with a fabricated example value. Combining
unreliable tool use with a model that's *also* producing a confident-looking
chain-of-thought would just make wrong answers more convincing, not less —
so this path is intentionally kept to what the model is actually reliable
at: showing its reasoning over what it already knows.

## Possible next steps (not implemented, good talking points)

- Swap the heuristic complexity classifier for a small fine-tuned/few-shot
  classifier, or an actual cheap-LLM-as-judge call.
- Add LlamaIndex-based RAG over real documents in place of the mock KB tool.
- Add an AutoGen-style critic agent that reviews the primary agent's answer
  before returning it for high-stakes tiers.
- Add tracing (e.g. LangSmith/OpenTelemetry) and a cost dashboard.
