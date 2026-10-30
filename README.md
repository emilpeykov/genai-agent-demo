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

## Possible next steps (not implemented, good talking points)

- Swap the heuristic complexity classifier for a small fine-tuned/few-shot
  classifier, or an actual cheap-LLM-as-judge call.
- Add LlamaIndex-based RAG over real documents in place of the mock KB tool.
- Add an AutoGen-style critic agent that reviews the primary agent's answer
  before returning it for high-stakes tiers.
- Add tracing (e.g. LangSmith/OpenTelemetry) and a cost dashboard.
